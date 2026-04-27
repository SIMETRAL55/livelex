"""
VAD processor wrapping Silero VAD for utterance boundary detection.

Reads audio chunks from an asyncio.Queue (produced by AudioCapture), maintains
a rolling buffer of recent audio, and fires two callbacks:
  - chunk_callback    — every chunk, regardless of speech/silence state
  - utterance_end_callback — when silence long enough after a speech segment

Silero VAD expects 512-sample frames at 16kHz.  Incoming chunks may be any
size (typically 1600 samples = 100ms at 16kHz), so this module buffers the
leftover sub-frame samples between calls.

Silero inference is synchronous (PyTorch forward pass).  It is offloaded to a
thread-pool executor via loop.run_in_executor so the asyncio event loop is
never blocked.
"""

import asyncio
import logging
from typing import Callable, Awaitable

import numpy as np
import torch

logger = logging.getLogger(__name__)

_VAD_FRAME_SAMPLES = 512  # Silero VAD requirement at 16kHz


class VADProcessor:
    """
    Silero-VAD-based speech/silence detector with rolling audio buffer.

    Usage::

        vad = VADProcessor()
        await vad.process(
            audio_queue,
            chunk_callback=on_chunk,
            utterance_end_callback=on_utterance_end,
        )
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_ms: int = 500,
        min_speech_ms: int = 250,
        buffer_max_seconds: int = 10,
    ) -> None:
        """
        Load the Silero VAD model and initialise state.

        Args:
            sample_rate: Sample rate of incoming audio (must be 16000 or 8000
                         for Silero; pipeline always uses 16000).
            threshold: Speech probability above which a frame is considered speech.
            min_silence_ms: Consecutive silence milliseconds required before an
                            utterance_end_callback is fired.
            min_speech_ms: Minimum speech duration (ms) in the current segment
                           before silence can trigger an utterance end.
            buffer_max_seconds: Maximum number of seconds kept in the rolling
                                buffer.  Older samples are discarded from the front.
        """
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._min_silence_s: float = min_silence_ms / 1000.0
        self._min_speech_s: float = min_speech_ms / 1000.0
        self._buffer_max_samples: int = sample_rate * buffer_max_seconds

        # Rolling buffer — grows as chunks arrive, trimmed from the front
        self._rolling_buffer: np.ndarray = np.zeros(0, dtype=np.float32)

        # VAD state (publicly readable via properties if needed)
        self.is_speech: bool = False
        self.silence_duration: float = 0.0
        self.speech_duration: float = 0.0

        # Leftover samples that did not fill a complete 512-sample frame
        self._remainder: np.ndarray = np.zeros(0, dtype=np.float32)

        # Accumulates audio during an utterance; emitted as one chunk on utterance end
        self._utterance_buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._in_utterance: bool = False

        # Load Silero VAD — pip package (silero-vad >= 5.x)
        try:
            from silero_vad import load_silero_vad  # type: ignore[import]

            self._model = load_silero_vad()
        except ImportError as exc:
            raise ImportError(
                "silero-vad is required.  Install it with:\n"
                "  pip install silero-vad\n"
                f"Original error: {exc}"
            ) from exc

        self._model.eval()
        logger.info(
            "Silero VAD loaded (threshold=%.2f  min_silence=%.0fms  min_speech=%.0fms)",
            threshold,
            min_silence_ms,
            min_speech_ms,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_buffer(self) -> np.ndarray:
        """
        Return a copy of the current rolling buffer.

        Returns:
            float32 numpy array containing the most recent audio, up to
            buffer_max_seconds long.  Shape: (N,).  Callers receive a copy
            so they cannot mutate internal state.
        """
        return self._rolling_buffer.copy()

    def reset_buffer(self) -> None:
        """
        Clear the rolling buffer.

        Called by streaming.py after it has flushed a completed utterance.
        Does not affect VAD state (is_speech, durations) — those are managed
        by the process() loop.

        Args:
            (none)

        Returns:
            None
        """
        self._rolling_buffer = np.zeros(0, dtype=np.float32)
        logger.debug("Rolling buffer cleared")

    async def process(
        self,
        audio_queue: asyncio.Queue,
        chunk_callback: Callable[[np.ndarray], Awaitable[None]],
        utterance_end_callback: Callable[[], Awaitable[None]],
        utterance_audio_callback: Callable[[np.ndarray], Awaitable[None]] | None = None,
    ) -> None:
        """
        Main processing loop.  Reads audio chunks from audio_queue indefinitely.

        For every chunk received:
          1. Appends it to the rolling buffer (trimming the front if over the
             max-length limit).
          2. Calls chunk_callback(chunk) for each chunk while speech is active,
             so the local streaming transcriber's poller always has fresh audio.
          3. Splits the chunk (plus any leftover sub-frame samples) into 512-sample
             frames, runs Silero VAD on each frame via a thread-pool executor, and
             updates the speech/silence state machine.
          4. When silence_duration ≥ min_silence_ms AND speech_duration ≥ min_speech_ms,
             calls utterance_audio_callback (if provided) with the full accumulated
             utterance buffer, then calls utterance_end_callback().

        This coroutine runs until cancelled.  It does not return normally.

        Args:
            audio_queue: asyncio.Queue producing float32 numpy arrays from AudioCapture.
            chunk_callback: Async callable invoked with each 100ms chunk during speech.
                            Signature: async def on_chunk(chunk: np.ndarray) -> None
            utterance_end_callback: Async callable invoked when an utterance ends.
                                    Signature: async def on_utterance_end() -> None
            utterance_audio_callback: Optional async callable invoked once per utterance
                                      end with the full concatenated speech buffer.
                                      Signature: async def on_utterance_audio(buf: np.ndarray) -> None

        Returns:
            None (runs until cancelled)
        """
        loop = asyncio.get_running_loop()
        frame_duration_s: float = _VAD_FRAME_SAMPLES / self._sample_rate

        while True:
            chunk: np.ndarray = await audio_queue.get()

            # ── 1. Update rolling buffer ──────────────────────────────
            self._rolling_buffer = np.concatenate([self._rolling_buffer, chunk])
            if len(self._rolling_buffer) > self._buffer_max_samples:
                self._rolling_buffer = self._rolling_buffer[-self._buffer_max_samples :]

            # ── 2. Run VAD on 512-sample frames ───────────────────────
            # Prepend any leftover sub-frame samples from the previous chunk
            to_process = (
                np.concatenate([self._remainder, chunk])
                if len(self._remainder)
                else chunk.copy()
            )

            offset = 0
            while offset + _VAD_FRAME_SAMPLES <= len(to_process):
                frame = to_process[offset : offset + _VAD_FRAME_SAMPLES].copy()
                offset += _VAD_FRAME_SAMPLES

                # Silero inference is a synchronous PyTorch call — offload to
                # thread pool so the event loop is never blocked.
                tensor = torch.from_numpy(frame)
                prob: float = await loop.run_in_executor(
                    None, self._infer, tensor
                )

                logger.debug("VAD frame prob=%.3f  is_speech=%s", prob, self.is_speech)

                if prob > self._threshold:
                    # ── Speech frame ──────────────────────────────────
                    self.speech_duration += frame_duration_s
                    self.silence_duration = 0.0
                    self.is_speech = True
                    self._in_utterance = True
                else:
                    # ── Silence frame ─────────────────────────────────
                    self.silence_duration += frame_duration_s

                    if (
                        self.silence_duration >= self._min_silence_s
                        and self.speech_duration >= self._min_speech_s
                    ):
                        logger.debug(
                            "Utterance end detected (speech=%.2fs  silence=%.2fs)",
                            self.speech_duration,
                            self.silence_duration,
                        )
                        if utterance_audio_callback is not None and len(self._utterance_buffer) > 0:
                            logger.debug(
                                "Emitting utterance buffer (%.2fs)",
                                len(self._utterance_buffer) / self._sample_rate,
                            )
                            await utterance_audio_callback(self._utterance_buffer.copy())
                        await utterance_end_callback()
                        self.is_speech = False
                        self.speech_duration = 0.0
                        self.silence_duration = 0.0
                        self._in_utterance = False
                        self._utterance_buffer = np.zeros(0, dtype=np.float32)

            # Save sub-frame remainder for next chunk
            self._remainder = to_process[offset:].copy()

            # Emit chunk and accumulate into utterance buffer while speech is active
            if self._in_utterance:
                self._utterance_buffer = np.concatenate([self._utterance_buffer, chunk])
                await chunk_callback(chunk)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer(self, tensor: torch.Tensor) -> float:
        """
        Run a single Silero VAD forward pass.

        Intended to be called from a thread-pool executor so it does not
        block the asyncio event loop.

        Args:
            tensor: 1-D float32 torch.Tensor of exactly 512 samples.

        Returns:
            Speech probability in [0.0, 1.0].
        """
        with torch.no_grad():
            prob = self._model(tensor, self._sample_rate)
        return float(prob)


# ---------------------------------------------------------------------------
# __main__ test block — exempt from no-print rule (CLAUDE.md §Hard Rules)
# Run with: python -m audio.vad --test [--input <wav_file>]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import asyncio
    import os
    import wave

    parser = argparse.ArgumentParser(
        description="LiveRex VAD test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Feeds a WAV file through VADProcessor in 100ms chunks and prints\n"
            "detected speech segments and utterance-end signals.\n"
            "Default input: debug/capture_test.wav"
        ),
    )
    parser.add_argument("--test", action="store_true", help="Run VAD test (required)")
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
        print("  python -m audio.vad --test --input path/to/audio.wav")
        raise SystemExit(1)

    # ── Load WAV ─────────────────────────────────────────────────────
    print("=" * 60)
    print("LiveRex — VAD Test")
    print("=" * 60)
    print()
    print(f"Input: {args.input}")

    with wave.open(args.input, "r") as wf:
        wav_rate = wf.getframerate()
        wav_channels = wf.getnchannels()
        total_frames = wf.getnframes()
        raw_bytes = wf.readframes(total_frames)

    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    if wav_channels > 1:
        audio_int16 = audio_int16.reshape(-1, wav_channels).mean(axis=1).astype(np.int16)
    audio_f32 = audio_int16.astype(np.float32) / 32768.0

    # Resample to 16kHz if needed
    if wav_rate != 16000:
        from fractions import Fraction
        from scipy.signal import resample_poly

        ratio = Fraction(16000, wav_rate).limit_denominator(1000)
        audio_f32 = resample_poly(audio_f32, ratio.numerator, ratio.denominator).astype(np.float32)
        print(f"Resampled from {wav_rate}Hz → 16000Hz")

    duration_s: float = len(audio_f32) / 16000
    print(f"Duration: {duration_s:.2f}s  ({len(audio_f32):,} samples at 16kHz)")
    print()

    # ── Split into 100ms chunks ───────────────────────────────────────
    CHUNK_SAMPLES = 1600  # 100ms at 16kHz
    chunks: list[np.ndarray] = [
        audio_f32[i : i + CHUNK_SAMPLES]
        for i in range(0, len(audio_f32), CHUNK_SAMPLES)
    ]
    # Pad last chunk to full size if needed
    if len(chunks[-1]) < CHUNK_SAMPLES:
        pad = np.zeros(CHUNK_SAMPLES - len(chunks[-1]), dtype=np.float32)
        chunks[-1] = np.concatenate([chunks[-1], pad])

    print(f"Processing {len(chunks)} chunks of 100ms each…")
    print()

    # ── Run VAD ───────────────────────────────────────────────────────
    async def _main() -> tuple[list, list, float]:
        """
        Feed all chunks through VADProcessor and collect results.

        Returns:
            Tuple of (speech_segments, utterance_ends, total_speech_s) where:
              speech_segments: list of (start_s, end_s) tuples
              utterance_ends:  list of (time_s, silence_s) tuples
              total_speech_s:  total seconds classified as speech
        """
        queue: asyncio.Queue = asyncio.Queue()
        vad = VADProcessor()

        # All mutable tracking state lives here so nested callbacks can
        # mutate it via simple list-element assignment (no nonlocal needed).
        speech_start: list[float | None] = [None]   # [0] = current segment start
        chunk_index: list[int] = [0]
        was_speech: list[bool] = [False]
        speech_segments: list[tuple[float, float]] = []
        utterance_ends: list[tuple[float, float]] = []
        total_speech_s: list[float] = [0.0]

        async def on_chunk(chunk: np.ndarray) -> None:
            t = chunk_index[0] * 0.1
            currently_speech = vad.is_speech

            if currently_speech and not was_speech[0]:
                speech_start[0] = t

            if not currently_speech and was_speech[0] and speech_start[0] is not None:
                seg_len = t - speech_start[0]
                speech_segments.append((speech_start[0], t))
                total_speech_s[0] += seg_len
                speech_start[0] = None

            was_speech[0] = currently_speech
            chunk_index[0] += 1

        async def on_utterance_end() -> None:
            t = chunk_index[0] * 0.1
            silence_s = vad.silence_duration
            utterance_ends.append((t, silence_s))
            print(f"  [END]     t={t:6.2f}s  silence={silence_s:.2f}s")

        process_task = asyncio.create_task(
            vad.process(queue, on_chunk, on_utterance_end)
        )

        for chunk in chunks:
            await queue.put(chunk)

        # Drain the queue before cancelling
        while not queue.empty():
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.2)

        process_task.cancel()
        try:
            await process_task
        except asyncio.CancelledError:
            pass

        # Close any speech segment still open at EOF
        if speech_start[0] is not None:
            end_t = chunk_index[0] * 0.1
            speech_segments.append((speech_start[0], end_t))
            total_speech_s[0] += end_t - speech_start[0]

        return speech_segments, utterance_ends, total_speech_s[0]

    speech_segments, utterance_ends, total_speech_s = asyncio.run(_main())

    print()
    print("Speech segments:")
    if speech_segments:
        for start, end in speech_segments:
            seg_len = end - start
            print(f"  [SPEECH]  t={start:6.2f}s → {end:6.2f}s  ({seg_len:.2f}s)")
    else:
        print("  (none detected — try lowering --threshold or check input audio)")

    print()
    print("Summary:")
    print(f"  Total audio    : {duration_s:.2f}s")
    print(f"  Speech time    : {total_speech_s:.2f}s  ({100 * total_speech_s / duration_s:.1f}%)")
    silence_s = duration_s - total_speech_s
    print(f"  Silence time   : {silence_s:.2f}s  ({100 * silence_s / duration_s:.1f}%)")
    print(f"  Utterance ends : {len(utterance_ends)}")
