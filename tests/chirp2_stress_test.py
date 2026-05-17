#!/usr/bin/env python3
"""
tests/chirp2_stress_test.py — Chirp 2 Stress Test Suite

Four suites run sequentially:
  1. Latency Under Load   — p50/p90/p95 latency from chunk send to result
  2. Session Stability    — 25-minute continuous run with realistic VAD patterns
  3. Reconnect / Recovery — forced gRPC channel close + recovery timing
  4. Audio Edge Cases     — silence, clipping, rapid-fire, long utterances

Usage:
  python tests/chirp2_stress_test.py

GCP credentials: ADC only (no API key).
Set GCP_PROJECT_ID, GCP_LOCATION, GCP_STT_RECOGNIZER via env vars or config.yaml [gcp] section.
Results logged to both terminal and tests/stress_results.log.
"""

import asyncio
import logging
import math
import os
import sys
import threading
import time
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ── Bootstrap: ensure project root is on sys.path ─────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transcription.chirp2 import Chirp2Transcriber  # noqa: E402
from utils.audio_utils import convert_to_pipeline_format  # noqa: E402

# ── Logging: terminal + file ───────────────────────────────────────────────────
LOG_FILE = ROOT / "tests" / "stress_results.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
_root = logging.getLogger()
_root.setLevel(logging.INFO)

_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_fmt)
_root.addHandler(_ch)

_fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_fh.setFormatter(_fmt)
_root.addHandler(_fh)

log = logging.getLogger("stress")

# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 1_600  # 100 ms at 16kHz

# ── Config loading ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    config_path = ROOT / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Apply gcp section to env vars if not already set
    gcp = cfg.get("gcp", {})
    for env_var, key in [
        ("GCP_PROJECT_ID", "project_id"),
        ("GCP_LOCATION", "location"),
        ("GCP_STT_RECOGNIZER", "recognizer"),
    ]:
        if gcp.get(key) and not os.environ.get(env_var):
            os.environ[env_var] = str(gcp[key])

    return cfg

# ── Audio helpers ──────────────────────────────────────────────────────────────

def load_or_generate_audio(seconds: float = 60.0) -> np.ndarray:
    """Return float32 16kHz mono audio of the requested length."""
    wav_path = ROOT / "debug" / "capture_test.wav"
    if wav_path.exists():
        with wave.open(str(wav_path), "r") as wf:
            raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32_768.0
            src_rate = wf.getframerate()
        if src_rate != SAMPLE_RATE:
            audio = convert_to_pipeline_format(audio, src_rate)
    else:
        # Synthesize: voiced tone with harmonics + syllabic AM + noise
        t = np.linspace(0.0, seconds, int(seconds * SAMPLE_RATE), dtype=np.float32)
        sig = np.zeros_like(t)
        for harmonic, amp in [(1, 0.30), (2, 0.20), (3, 0.15), (4, 0.10), (5, 0.07)]:
            sig += amp * np.sin(2 * math.pi * 180 * harmonic * t)
        am = 0.5 + 0.5 * np.sin(2 * math.pi * 3.2 * t)  # ~3 Hz syllable rhythm
        sig = (sig * am + 0.02 * np.random.randn(*sig.shape).astype(np.float32))
        audio = np.clip(sig, -1.0, 1.0)

    target = int(seconds * SAMPLE_RATE)
    if len(audio) < target:
        reps = math.ceil(target / len(audio))
        audio = np.tile(audio, reps)
    return audio[:target].copy()


def make_silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def make_clipped(seconds: float) -> np.ndarray:
    """Alternating ±1.0 samples — maximum amplitude."""
    n = int(seconds * SAMPLE_RATE)
    arr = np.ones(n, dtype=np.float32)
    arr[1::2] = -1.0
    return arr


def chunkify(audio: np.ndarray, pad: bool = True) -> list[np.ndarray]:
    """Split audio into CHUNK_SAMPLES slices, optionally padding the last one."""
    chunks = []
    for i in range(0, len(audio), CHUNK_SAMPLES):
        c = audio[i : i + CHUNK_SAMPLES]
        if pad and len(c) < CHUNK_SAMPLES:
            c = np.pad(c, (0, CHUNK_SAMPLES - len(c)))
        chunks.append(c)
    return chunks

# ── Statistics helpers ─────────────────────────────────────────────────────────

def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = max(0, min(int(len(s) * p / 100), len(s) - 1))
    return s[idx]


def fmt_ms(v: float) -> str:
    return "N/A" if math.isnan(v) else f"{v * 1000:.0f}ms"

# ── Suite formatting ───────────────────────────────────────────────────────────

def suite_header(n: int, title: str) -> None:
    log.info("")
    log.info("═" * 62)
    log.info("  SUITE %d: %s", n, title)
    log.info("═" * 62)


def suite_result(passed: bool, details: str = "") -> None:
    status = "PASS ✓" if passed else "FAIL ✗"
    log.info("  → %s  %s", status, details)
    log.info("")

# ── Feeder coroutine factory ───────────────────────────────────────────────────

async def feed_chunks(
    chunks: list[np.ndarray],
    q: asyncio.Queue,
    sleep_s: float = 0.100,
) -> None:
    """Push chunks into q with optional sleep between, then push sentinel None."""
    for chunk in chunks:
        await q.put(chunk)
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)
    await q.put(None)

# ═════════════════════════════════════════════════════════════════════════════
# SUITE 1 — Latency Under Load
# ═════════════════════════════════════════════════════════════════════════════

async def suite_latency(config: dict) -> bool:
    suite_header(1, "Latency Under Load")

    timing_lock = threading.Lock()
    chunk_times: list[float] = []   # monotonic time when each chunk was sent
    interim_times: list[float] = []
    final_times: list[float] = []

    def on_timing(event: str, _index: int, t: float) -> None:
        with timing_lock:
            if event == "chunk_sent":
                chunk_times.append(t)
            elif event == "interim":
                interim_times.append(t)
            elif event == "final":
                final_times.append(t)

    transcriber = Chirp2Transcriber(config["chirp2"], on_timing=on_timing)
    audio = load_or_generate_audio(60.0)
    chunks = chunkify(audio)
    log.info("  Audio: %ds / %d chunks at 100ms cadence", len(audio) // SAMPLE_RATE, len(chunks))

    q: asyncio.Queue = asyncio.Queue()
    feed_task = asyncio.create_task(feed_chunks(chunks, q, sleep_s=0.100))

    results_received = 0
    try:
        async for _token in transcriber.transcribe_stream(q):
            results_received += 1
    except Exception as exc:
        log.error("  Stream error: %s", exc)
        await feed_task
        suite_result(False, f"stream error: {exc}")
        return False

    await feed_task

    # For each result, latency = result_time − last chunk sent before it
    def compute_latencies(result_times: list[float]) -> list[float]:
        with timing_lock:
            ct = sorted(chunk_times)
        lats = []
        for rt in result_times:
            preceding = [c for c in ct if c <= rt]
            if preceding:
                lats.append(rt - preceding[-1])
        return lats

    interim_lats = compute_latencies(interim_times)
    final_lats = compute_latencies(final_times)

    # Count latency spikes: gap > 500ms between consecutive result events
    all_results = sorted(interim_times + final_times)
    spikes = sum(
        1 for i in range(1, len(all_results)) if all_results[i] - all_results[i - 1] > 0.500
    )

    log.info("  Results received: %d  (interim=%d  final=%d)",
             results_received, len(interim_times), len(final_times))
    log.info("  ┌────────────────────────────────────────────────────┐")
    log.info("  │ Metric              p50        p90        p95      │")
    log.info("  ├────────────────────────────────────────────────────┤")
    log.info("  │ Interim latency  %8s   %8s   %8s   │",
             fmt_ms(percentile(interim_lats, 50)),
             fmt_ms(percentile(interim_lats, 90)),
             fmt_ms(percentile(interim_lats, 95)))
    log.info("  │ Final latency    %8s   %8s   %8s   │",
             fmt_ms(percentile(final_lats, 50)),
             fmt_ms(percentile(final_lats, 90)),
             fmt_ms(percentile(final_lats, 95)))
    log.info("  │ Latency spikes (>500ms gap): %-4d                  │", spikes)
    log.info("  └────────────────────────────────────────────────────┘")

    passed = results_received > 0
    suite_result(passed, f"results={results_received} spikes={spikes}")
    return passed

# ═════════════════════════════════════════════════════════════════════════════
# SUITE 2 — Session Stability (25 minutes)
# ═════════════════════════════════════════════════════════════════════════════

async def suite_stability(config: dict) -> bool:
    suite_header(2, "Session Stability (25 minutes)")

    # Counters (modified only from async task — no lock needed here)
    sessions_opened = 0
    sessions_closed_clean = 0
    unexpected_disconnects = 0
    errors_by_code: dict[str, int] = defaultdict(int)
    last_latency_ms = 0.0

    # Error callback runs in the gRPC thread; use a lock for the shared dict
    error_lock = threading.Lock()

    def make_error_cb() -> Any:
        def on_error(exc: Exception) -> None:
            nonlocal unexpected_disconnects
            with error_lock:
                unexpected_disconnects += 1
                code = type(exc).__name__
                if hasattr(exc, "grpc_status_code") and exc.grpc_status_code is not None:
                    code = str(exc.grpc_status_code)
                errors_by_code[code] += 1
        return on_error

    TOTAL_SECONDS = 25 * 60
    SPEECH_DURATIONS = [7.0, 8.0, 6.0, 9.0, 5.0, 10.0, 7.0, 8.0]
    SILENCE_DURATIONS = [1.5, 2.0, 1.0, 2.5, 1.0, 1.5]

    base_audio = load_or_generate_audio(60.0)

    start_time = time.monotonic()
    next_heartbeat = start_time + 300.0
    utter_idx = 0
    silence_idx = 0

    log.info("  Starting 25-minute stability run (VAD sim: speech 5–10s / silence 1–2.5s)...")

    while True:
        now = time.monotonic()
        elapsed = now - start_time
        if elapsed >= TOTAL_SECONDS:
            break

        # Heartbeat every 5 minutes
        if now >= next_heartbeat:
            log.info("  [T+%.0fmin] sessions_opened=%d errors=%d last_latency=%.0fms",
                     elapsed / 60, sessions_opened, unexpected_disconnects, last_latency_ms)
            next_heartbeat += 300.0

        # Determine utterance length
        speech_dur = SPEECH_DURATIONS[utter_idx % len(SPEECH_DURATIONS)]
        utter_idx += 1
        remaining = TOTAL_SECONDS - (time.monotonic() - start_time)
        if remaining <= 0:
            break
        speech_dur = min(speech_dur, remaining)

        # Slice audio for this utterance
        n_samples = int(speech_dur * SAMPLE_RATE)
        audio_offset = (sessions_opened * CHUNK_SAMPLES * 30) % max(1, len(base_audio) - n_samples)
        speech = base_audio[audio_offset : audio_offset + n_samples]
        if len(speech) < n_samples:
            speech = np.tile(base_audio, math.ceil(n_samples / len(base_audio)))[:n_samples]

        q: asyncio.Queue = asyncio.Queue()
        transcriber = Chirp2Transcriber(config["chirp2"], on_error=make_error_cb())
        sessions_opened += 1
        session_start_t = time.monotonic()
        got_result = False

        feed_task = asyncio.create_task(feed_chunks(chunkify(speech), q, sleep_s=0.100))

        try:
            async for _token in transcriber.transcribe_stream(q):
                if not got_result:
                    last_latency_ms = (time.monotonic() - session_start_t) * 1000
                    got_result = True
            sessions_closed_clean += 1
        except Exception as exc:
            log.warning("  Session %d raised: %s", sessions_opened, exc)
            with error_lock:
                unexpected_disconnects += 1
                errors_by_code[type(exc).__name__] += 1

        await feed_task

        # Silence gap
        silence_dur = SILENCE_DURATIONS[silence_idx % len(SILENCE_DURATIONS)]
        silence_idx += 1
        remaining = TOTAL_SECONDS - (time.monotonic() - start_time)
        if remaining <= 0:
            break
        await asyncio.sleep(min(silence_dur, remaining))

    total_runtime = time.monotonic() - start_time
    error_rate = unexpected_disconnects / max(1, sessions_opened)

    log.info("  ── Stability Report ──────────────────────────────")
    log.info("  Total runtime:           %.1f min", total_runtime / 60)
    log.info("  Sessions opened:         %d", sessions_opened)
    log.info("  Sessions closed cleanly: %d", sessions_closed_clean)
    log.info("  Unexpected disconnects:  %d", unexpected_disconnects)
    log.info("  Error rate:              %.1f%%", error_rate * 100)
    log.info("  Errors by code:          %s",
             dict(errors_by_code) if errors_by_code else "(none)")
    log.info("  Any unrecovered failure: %s",
             "No" if sessions_closed_clean + unexpected_disconnects >= sessions_opened else "Yes")

    passed = sessions_opened > 0 and error_rate < 0.10
    suite_result(passed,
                 f"sessions={sessions_opened} errors={unexpected_disconnects} "
                 f"error_rate={error_rate:.1%}")
    return passed

# ═════════════════════════════════════════════════════════════════════════════
# SUITE 3 — Reconnect / Recovery
# ═════════════════════════════════════════════════════════════════════════════

def _force_close_channel(transcriber: Chirp2Transcriber) -> None:
    """Best-effort: close the underlying gRPC channel to simulate a network drop."""
    client = transcriber._client
    try:
        # GAPIC transports expose _grpc_channel or close()
        transport = getattr(client, "_transport", None)
        if transport is None:
            transport = getattr(client, "transport", None)
        if transport is not None:
            channel = getattr(transport, "_grpc_channel", None)
            if channel is not None:
                channel.close()
                return
            close_fn = getattr(transport, "close", None)
            if close_fn is not None:
                close_fn()
                return
        # Final fallback
        close_fn = getattr(client, "close", None)
        if close_fn is not None:
            close_fn()
    except Exception as exc:
        log.warning("  _force_close_channel: %s", exc)


async def suite_reconnect(config: dict) -> bool:
    suite_header(3, "Reconnect / Recovery")

    TRIALS = 5
    NORMAL_SECS = 10.0
    RECOVERY_SECS = 15.0
    THRESHOLD = 3.0

    base_audio = load_or_generate_audio(30.0)
    recovery_times: list[float] = []
    successes = 0

    for trial in range(1, TRIALS + 1):
        log.info("  Trial %d/%d — streaming %ds normally...", trial, TRIALS, int(NORMAL_SECS))

        # Phase 1: normal stream
        t1 = Chirp2Transcriber(config["chirp2"])
        normal_chunks = chunkify(base_audio[: int(NORMAL_SECS * SAMPLE_RATE)])
        q1: asyncio.Queue = asyncio.Queue()
        f1 = asyncio.create_task(feed_chunks(normal_chunks, q1, sleep_s=0.100))
        try:
            async for _ in t1.transcribe_stream(q1):
                pass
        except Exception:
            pass
        await f1

        # Phase 2: force disconnect
        log.info("  Trial %d/%d — forcing gRPC channel close...", trial, TRIALS)
        disconnect_t = time.monotonic()
        _force_close_channel(t1)

        # Phase 3: immediately open new session and measure recovery
        log.info("  Trial %d/%d — starting recovery stream...", trial, TRIALS)
        t2 = Chirp2Transcriber(config["chirp2"])
        recovery_audio = base_audio[: int(RECOVERY_SECS * SAMPLE_RATE)]
        q2: asyncio.Queue = asyncio.Queue()
        f2 = asyncio.create_task(feed_chunks(chunkify(recovery_audio), q2, sleep_s=0.100))

        first_result_t: float | None = None
        try:
            async for _tok in t2.transcribe_stream(q2):
                if first_result_t is None:
                    first_result_t = time.monotonic()
        except Exception as exc:
            log.error("  Trial %d: recovery stream error: %s", trial, exc)

        await f2

        if first_result_t is not None:
            rt = first_result_t - disconnect_t
            recovery_times.append(rt)
            successes += 1
            flag = " [EXCEEDED 3s]" if rt > THRESHOLD else ""
            log.info("  Trial %d: RECOVERED in %.2fs%s", trial, rt, flag)
        else:
            log.info("  Trial %d: FAILED — no results received after forced disconnect", trial)

        if trial < TRIALS:
            log.info("  Waiting 15s between trials...")
            await asyncio.sleep(15.0)

    mean_rt = sum(recovery_times) / len(recovery_times) if recovery_times else float("nan")
    exceeded = sum(1 for t in recovery_times if t > THRESHOLD)

    log.info("  ── Reconnect Report ──────────────────────────────")
    log.info("  Success rate:        %d/%d", successes, TRIALS)
    log.info("  Mean recovery time:  %.2fs", mean_rt)
    log.info("  Exceeded 3s:         %d/%d", exceeded, successes)

    passed = successes >= math.ceil(TRIALS * 0.8)
    suite_result(passed, f"success={successes}/{TRIALS} mean_recovery={mean_rt:.2f}s")
    return passed

# ═════════════════════════════════════════════════════════════════════════════
# SUITE 4 — Audio Edge Cases
# ═════════════════════════════════════════════════════════════════════════════

async def run_edge_case(
    chirp_config: dict,
    label: str,
    chunks: list[np.ndarray],
    sleep_s: float = 0.100,
) -> tuple[bool, str, int]:
    """
    Returns (no_exception, exception_str, token_count).
    """
    transcriber = Chirp2Transcriber(chirp_config)
    q: asyncio.Queue = asyncio.Queue()
    feed_task = asyncio.create_task(feed_chunks(chunks, q, sleep_s=sleep_s))
    token_count = 0
    exc_str = ""
    try:
        async for _tok in transcriber.transcribe_stream(q):
            token_count += 1
    except Exception as exc:
        exc_str = str(exc)
    await feed_task
    return (exc_str == ""), exc_str, token_count


async def suite_edge_cases(config: dict) -> bool:
    suite_header(4, "Audio Edge Cases")

    base_audio = load_or_generate_audio(50.0)

    # Build sub-cases: (label, chunks, sleep_s)
    cases: list[tuple[str, list[np.ndarray], float]] = []

    # 1. 100ms pure silence — single chunk
    cases.append(("100ms pure silence (1 chunk)", [make_silence(0.1)], 0.100))

    # 2. 30s pure silence — continuous
    cases.append(("30s pure silence", chunkify(make_silence(30.0)), 0.100))

    # 3. 5s maximum amplitude (clipped)
    cases.append(("5s clipped audio (all samples = ±1.0)", chunkify(make_clipped(5.0)), 0.100))

    # 4. 10×200ms utterances with 100ms silence gaps
    short_speech = chunkify(base_audio[: CHUNK_SAMPLES * 2])  # ~200ms
    silence_chunk = [make_silence(0.1)]
    mixed: list[np.ndarray] = []
    for _ in range(10):
        mixed.extend(short_speech)
        mixed.extend(silence_chunk)
    cases.append(("10×200ms utterances, 100ms gaps", mixed, 0.100))

    # 5. 45s continuous utterance (near 4-min session limit)
    cases.append(("45s continuous (near session cap)", chunkify(base_audio[: int(45 * SAMPLE_RATE)]), 0.100))

    # 6. 500 chunks sent as fast as possible (no sleep)
    rapid_chunks = chunkify(base_audio[: CHUNK_SAMPLES * 500])[:500]
    cases.append(("500 chunks rapid-fire (no sleep)", rapid_chunks, 0.0))

    all_passed = True
    log.info("  %-47s  %-4s  %-6s  %s", "Sub-case", "Pass", "Tokens", "Exception")
    log.info("  " + "─" * 80)

    for label, chunks, sleep_s in cases:
        ok, exc_str, n_tokens = await run_edge_case(config["chirp2"], label, chunks, sleep_s)
        status = "Y" if ok else "N"
        exc_display = (exc_str[:38] + "…") if len(exc_str) > 39 else (exc_str or "(none)")
        log.info("  %-47s  %-4s  %-6d  %s", label[:47], status, n_tokens, exc_display)
        if not ok:
            all_passed = False

    suite_result(all_passed,
                 "all sub-cases completed without exception"
                 if all_passed else "one or more sub-cases raised an exception")
    return all_passed

# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  Chirp 2 Stress Test  —  %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("══════════════════════════════════════════════════════════════")

    config = load_config()

    project_id = os.environ.get("GCP_PROJECT_ID", "")
    location   = os.environ.get("GCP_LOCATION", "asia-northeast1")
    recognizer = os.environ.get("GCP_STT_RECOGNIZER", "")

    log.info("  GCP project_id : %s", project_id  or "(not set — will fail)")
    log.info("  GCP location   : %s", location)
    log.info("  GCP recognizer : %s", recognizer or "(not set — will fail)")
    log.info("  Auth           : ADC (Application Default Credentials)")
    log.info("  Log file       : %s", LOG_FILE)

    if not project_id or not recognizer:
        log.error("")
        log.error("  ABORT: GCP_PROJECT_ID and GCP_STT_RECOGNIZER must be set.")
        log.error("  Add them to config.yaml under the [gcp] section, or export as env vars.")
        print("STRESS TEST: FAIL (missing GCP credentials)")
        sys.exit(1)

    results: list[tuple[str, bool]] = []

    r1 = await suite_latency(config)
    results.append(("Suite 1 — Latency Under Load", r1))

    r2 = await suite_stability(config)
    results.append(("Suite 2 — Session Stability", r2))

    r3 = await suite_reconnect(config)
    results.append(("Suite 3 — Reconnect/Recovery", r3))

    r4 = await suite_edge_cases(config)
    results.append(("Suite 4 — Audio Edge Cases", r4))

    log.info("══════════════════════════════════════════════════════════════")
    log.info("  FINAL RESULTS")
    log.info("══════════════════════════════════════════════════════════════")
    for name, passed in results:
        log.info("  [%s] %s", "PASS" if passed else "FAIL", name)

    failed = sum(1 for _, p in results if not p)
    verdict = "STRESS TEST: PASS" if failed == 0 else f"STRESS TEST: FAIL ({failed} suite{'s' if failed != 1 else ''} failed)"
    log.info(verdict)
    print(verdict)


if __name__ == "__main__":
    asyncio.run(main())
