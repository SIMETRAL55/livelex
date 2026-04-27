"""
Local transcription backend using faster-whisper + kotoba-whisper-v2.0.

Wraps faster-whisper's WhisperModel as an AbstractTranscriber.  The model is
loaded once at construction time (HuggingFace download on first run; cached
locally on subsequent runs).  Each transcribe_sync call runs a full batch
transcription on the supplied audio and returns a list of Word objects with
buffer-relative timestamps.

CPU usage note
--------------
This module defaults to compute_type="int8" and device="cpu" as set in
config.yaml.  float16 requires a CUDA device — do not change that value
without a GPU.  int8 inference on CPU is ~3-4x slower than float16 on GPU
but produces equivalent accuracy for short utterances.
"""

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import asyncio
import numpy as np

from transcription.base import AbstractTranscriber, Word

logger = logging.getLogger(__name__)


class LocalTranscriber(AbstractTranscriber):
    """
    faster-whisper / kotoba-whisper-v2.0 transcription backend.

    Usage::

        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        transcriber = LocalTranscriber(cfg["local"])
        words = transcriber.transcribe_sync(audio_array, language="en")
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Load the faster-whisper model from HuggingFace (or local cache).

        On first run this downloads kotoba-whisper-v2.0 (~750MB for int8).
        Subsequent runs load from the faster-whisper cache directory
        (~/.cache/huggingface/hub/).

        Args:
            config: Dict matching the ``local`` section of config.yaml.
                    Required keys:
                      model           (str)  — HuggingFace model identifier
                      compute_type    (str)  — "int8", "float16", "float32", etc.
                      device          (str)  — "cpu" or "cuda"
                      beam_size       (int)  — greedy-search beam width
                      flush_beam_size (int)  — beam width used on utterance flush

        Raises:
            KeyError: If a required key is missing from config.
            RuntimeError: If faster-whisper cannot load the model (e.g. CUDA
                          requested but not available).
        """
        from faster_whisper import WhisperModel  # type: ignore[import]

        self._model_name: str = config["model"]
        self._compute_type: str = config["compute_type"]
        self._device: str = config["device"]
        self._beam_size: int = int(config["beam_size"])
        self._flush_beam_size: int = int(config["flush_beam_size"])

        logger.info(
            "Loading faster-whisper model: %s  (device=%s  compute_type=%s) — "
            "first run may take several minutes while the model downloads",
            self._model_name,
            self._device,
            self._compute_type,
        )

        t0 = time.perf_counter()
        self._model = WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
        )
        load_ms = (time.perf_counter() - t0) * 1000
        logger.info("Model loaded in %.0fms", load_ms)

    def transcribe_sync(
        self,
        audio: np.ndarray,
        language: str = "en",
        *,
        beam_size: int | None = None,
        initial_prompt: str | None = None,
    ) -> list[Word]:
        """
        Transcribe a complete audio buffer and return per-word timing.

        Calls faster-whisper's transcribe() with word_timestamps=True,
        exhausts the segment generator, and returns a list of Word objects
        with buffer-relative timestamps.

        Args:
            audio: float32 numpy array, shape (N,).  Must be 16kHz mono,
                   values in [-1.0, 1.0].  Arrays shorter than ~0.1s may
                   produce empty output.
            language: BCP-47 language code passed to faster-whisper.
                      Default "en" (english).
            beam_size: Override beam width for this call.  If None, uses
                       self._beam_size from config (1 for streaming calls).
            initial_prompt: Priming text passed to the Whisper decoder to
                            stabilise output across consecutive ASR calls.
                            Typically the last ~200 chars of committed text.

        Returns:
            List of Word objects with buffer-relative timestamps.  Empty list
            if no words were recognised.
        """
        effective_beam = beam_size if beam_size is not None else self._beam_size

        logger.debug(
            "prompt: %r",
            (initial_prompt or "")[-60:],
        )

        t0 = time.perf_counter()
        segments, _info = self._model.transcribe(
            audio,
            language=language,
            beam_size=effective_beam,
            word_timestamps=True,
            initial_prompt=initial_prompt,
        )
        words: list[Word] = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    words.append(Word(text=w.word, start=w.start, end=w.end))

        latency_ms = (time.perf_counter() - t0) * 1000
        text_preview = "".join(w.text for w in words).strip()

        logger.debug(
            "ASR latency: %.0fms | beam=%d | len=%ds | words=%d | text=%r",
            latency_ms,
            effective_beam,
            len(audio) // 16000,
            len(words),
            text_preview[:60],
        )

        if not words:
            logger.debug(
                "ASR returned no words for %.1fs of audio (language=%s)",
                len(audio) / 16000,
                language,
            )

        return words

    async def transcribe_stream(
        self, audio_queue: asyncio.Queue
    ) -> AsyncIterator[str]:
        """
        Not implemented for the local backend.

        The local model is synchronous and not suited to streaming token
        output.  Streaming is handled by the polling loop in streaming.py
        which calls transcribe_sync repeatedly.  The Gemini backend
        implements this method for true streaming.

        Args:
            audio_queue: Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "LocalTranscriber does not support transcribe_stream(). "
            "Use transcribe_sync() inside a polling loop (see streaming.py)."
        )
        yield  # satisfy AsyncIterator protocol for type checkers


# ---------------------------------------------------------------------------
# __main__ test block — exempt from no-print rule (CLAUDE.md §Hard Rules)
# Run with: python -m transcription.local --test [--input <wav_file>]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os
    import wave

    import numpy as np
    import yaml  # type: ignore[import]

    parser = argparse.ArgumentParser(
        description="LiveRex local transcription test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Loads a WAV file and transcribes it twice with kotoba-whisper-v2.0.\n"
            "First call includes model-load time; second call shows true inference latency.\n"
            "Default input: debug/capture_test.wav"
        ),
    )
    parser.add_argument("--test", action="store_true", help="Run transcription test (required)")
    parser.add_argument(
        "--input",
        default="debug/capture_test.wav",
        metavar="WAV_FILE",
        help="Path to input WAV file (default: debug/capture_test.wav)",
    )
    args = parser.parse_args()

    if not args.test:
        parser.print_help()
        raise SystemExit(1)

    if not os.path.exists(args.input):
        print(f"ERROR: WAV file not found: {args.input}")
        print()
        print("Run the audio capture test first to generate a test file:")
        print("  python -m audio.capture --test")
        print()
        print("Or point to an existing WAV file:")
        print("  python -m transcription.local --test --input path/to/audio.wav")
        raise SystemExit(1)

    # ── Load config ───────────────────────────────────────────────────
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path) as f:
        full_config = yaml.safe_load(f)
    local_config = full_config["local"]
    language = full_config["transcription"]["language"]

    print("=" * 60)
    print("LiveRex — Local Transcription Test")
    print("=" * 60)
    print()
    print(f"Model        : {local_config['model']}")
    print(f"Device       : {local_config['device']}")
    print(f"Compute type : {local_config['compute_type']}")
    print(f"Language     : {language}")
    print(f"Input        : {args.input}")
    print()

    # ── Load WAV ─────────────────────────────────────────────────────
    with wave.open(args.input, "r") as wf:
        wav_rate = wf.getframerate()
        wav_channels = wf.getnchannels()
        total_frames = wf.getnframes()
        raw_bytes = wf.readframes(total_frames)

    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    if wav_channels > 1:
        audio_int16 = audio_int16.reshape(-1, wav_channels).mean(axis=1).astype(np.int16)
    audio_f32 = audio_int16.astype(np.float32) / 32768.0

    if wav_rate != 16000:
        from fractions import Fraction
        from scipy.signal import resample_poly  # type: ignore[import]

        ratio = Fraction(16000, wav_rate).limit_denominator(1000)
        audio_f32 = resample_poly(audio_f32, ratio.numerator, ratio.denominator).astype(np.float32)
        print(f"Resampled from {wav_rate}Hz → 16000Hz")

    duration_s = len(audio_f32) / 16000
    print(f"Audio duration : {duration_s:.2f}s  ({len(audio_f32):,} samples)")
    print()

    # ── Load model ────────────────────────────────────────────────────
    print("Loading model (first run downloads from HuggingFace — may take several minutes)…")
    t_load_start = time.perf_counter()
    transcriber = LocalTranscriber(local_config)
    load_ms = (time.perf_counter() - t_load_start) * 1000
    print(f"Model loaded in {load_ms:.0f}ms")
    print()

    # ── First transcription (warm-up) ─────────────────────────────────
    print("Run 1 (includes any JIT warm-up)…")
    t0 = time.perf_counter()
    words1 = transcriber.transcribe_sync(audio_f32, language=language)
    latency1_ms = (time.perf_counter() - t0) * 1000

    if words1:
        text1 = "".join(w.text for w in words1).strip()
        last_end1 = words1[-1].end
        print(f"  Transcript : {text1}")
        print(f"  Last word  : {last_end1:.2f}s")
    else:
        print("  Transcript : (empty — model returned no segments)")
        print("  → Check that the WAV contains speech and language is set correctly")
    print(f"  Latency    : {latency1_ms:.0f}ms")
    print()

    # ── Second transcription (true inference latency) ─────────────────
    print("Run 2 (true inference latency, with prompt priming)…")
    prompt = "".join(w.text for w in words1)[-200:] if words1 else None
    t0 = time.perf_counter()
    words2 = transcriber.transcribe_sync(audio_f32, language=language, initial_prompt=prompt)
    latency2_ms = (time.perf_counter() - t0) * 1000

    if words2:
        text2 = "".join(w.text for w in words2).strip()
        last_end2 = words2[-1].end
        print(f"  Transcript : {text2}")
        print(f"  Last word  : {last_end2:.2f}s")
    else:
        print("  Transcript : (empty)")
    print(f"  Latency    : {latency2_ms:.0f}ms")
    print()

    # ── Summary ───────────────────────────────────────────────────────
    print("Summary:")
    print(f"  Audio duration  : {duration_s:.2f}s")
    print(f"  Run 1 latency   : {latency1_ms:.0f}ms  (includes warm-up)")
    print(f"  Run 2 latency   : {latency2_ms:.0f}ms  (true inference)")
    rtf = latency2_ms / (duration_s * 1000)
    print(f"  Real-time factor: {rtf:.2f}x  (< 1.0 = faster than real time)")
