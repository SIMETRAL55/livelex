"""
Abstract base class and shared types for all transcription backends.

Both the local (faster-whisper) and API (Gemini) backends implement this
interface so the rest of the pipeline can swap between them with a single
config change.

Audio contract (must be satisfied by callers before passing audio here):
  - Sample rate : 16 000 Hz
  - Channels    : 1 (mono)
  - dtype       : float32
  - Range       : -1.0 to 1.0

Any audio that doesn't match must be converted in utils/audio_utils.py
before reaching a transcriber.
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Word:
    """
    A single recognised word with timing information.

    Timestamps are buffer-relative when returned by transcribe_sync, and
    are converted to absolute stream time by the streaming layer
    (by adding buffer_anchor_s).

    Attributes:
        text:  Word text as returned by the ASR engine; typically includes
               a leading space (e.g. " hello").
        start: Start time in seconds relative to the start of the audio
               buffer passed to transcribe_sync (or absolute stream time
               after anchor conversion in streaming.py).
        end:   End time in seconds (same reference frame as start).
    """

    text: str
    start: float
    end: float


class AbstractTranscriber:
    """
    Common interface for synchronous batch transcription and async streaming.

    Subclasses must override both methods.  The default implementations raise
    NotImplementedError so missing overrides are caught at runtime rather than
    silently returning wrong results.
    """

    native_streaming: bool = False
    """True when the backend supports transcribe_stream() (Gemini only)."""

    def transcribe_sync(
        self,
        audio: np.ndarray,
        language: str = "en",
        *,
        beam_size: int | None = None,
        initial_prompt: str | None = None,
    ) -> list[Word]:
        """
        Transcribe a complete audio buffer synchronously.

        Intended for the polling loop in streaming.py: called every 300ms on
        the full rolling buffer.  Must return quickly enough not to stall the
        pipeline — see latency targets below.

        Args:
            audio: float32 numpy array, shape (N,).  Must be 16kHz mono,
                   values in [-1.0, 1.0].  Minimum useful length is roughly
                   0.5s (8 000 samples); very short arrays may return an
                   empty list.
            language: BCP-47 language code passed to the ASR engine.
                      Examples: "ja" (Japanese), "en" (English), "ko" (Korean).
            beam_size: Override beam width for this call.  None uses backend
                       default (typically 1 for streaming speed).
            initial_prompt: Text to prime the Whisper decoder; the last ~200
                            characters of already-committed transcript.  Reduces
                            hallucination and stabilises consecutive outputs.

        Returns:
            List of Word objects in speech order.  Each Word carries buffer-
            relative timestamps (seconds from the start of `audio`).  Returns
            an empty list if the model produced no output.

        Raises:
            NotImplementedError: If the subclass has not overridden this method.

        Latency targets (P90 on reference hardware):
            Local model (CPU int8)  : < 2 000ms for a 5s utterance
            Local model (CUDA fp16) : <   200ms for a 5s utterance
            Gemini Live API         : <   400ms (first token)
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement transcribe_sync()"
        )

    async def transcribe_stream(
        self, audio_queue: asyncio.Queue
    ) -> AsyncIterator[str]:
        """
        Stream transcription tokens as they are produced (Gemini backend).

        Consumes audio chunks from audio_queue and yields text tokens as the
        backend produces them.  Designed for the Gemini Live API which returns
        partial tokens in real time; local backends do not use this method.

        Args:
            audio_queue: asyncio.Queue producing float32 numpy arrays from
                         AudioCapture (same format as transcribe_sync).

        Yields:
            Partial or complete text tokens as strings, in the order they
            are produced by the backend.  May yield empty strings; callers
            should skip them.

        Raises:
            NotImplementedError: If the subclass has not overridden this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement transcribe_stream()"
        )
        # Satisfy the AsyncIterator protocol even on the error path so type
        # checkers don't complain about a missing yield.
        yield  # type: ignore[misc]
