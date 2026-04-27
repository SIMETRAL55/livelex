"""
Google Cloud Speech-to-Text V2 (Chirp 2) transcription backend.
Refined for VAD-driven ephemeral streaming.
Uses threading.Queue as a bridge to avoid run_coroutine_threadsafe races.
"""

import asyncio
import logging
import os
import time
import queue as thread_queue
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech

from transcription.base import AbstractTranscriber, Word
from utils.audio_utils import float32_to_pcm16

logger = logging.getLogger(__name__)

_MIN_AUDIO_SECONDS: float = 0.1
_SAMPLE_RATE: int = 16000


class Chirp2Transcriber(AbstractTranscriber):
    """
    Chirp 2 transcription backend using Google Cloud Speech-to-Text V2.
    """

    native_streaming: bool = True

    def __init__(self, config: dict[str, Any]) -> None:
        self._project_id = os.getenv("GCP_PROJECT_ID", "")
        self._location = os.getenv("GCP_LOCATION", "asia-northeast1")
        self._recognizer_path = os.getenv("GCP_STT_RECOGNIZER", "")
        self._model = config["model"]

        if not self._project_id or not self._recognizer_path:
            logger.warning("GCP_PROJECT_ID or GCP_STT_RECOGNIZER not set in environment.")

        client_options = ClientOptions(
            api_endpoint=f"{self._location}-speech.googleapis.com"
        )
        # Using sync client for more predictable behavior in this environment
        self._client = speech_v2.SpeechClient(client_options=client_options)

        logger.info(
            "Chirp2Transcriber ready (project=%s  location=%s  model=%s)",
            self._project_id,
            self._location,
            self._model,
        )

    def transcribe_sync(self, audio: np.ndarray, language="en", **kwargs) -> list[Word]:
        return []

    async def transcribe_stream(self, audio_queue: asyncio.Queue) -> AsyncIterator[str]:
        """
        Runs an ephemeral Chirp 2 gRPC session.
        Uses threading.Queue as bridge to avoid run_coroutine_threadsafe races.
        """
        output_queue: asyncio.Queue[str | None] = asyncio.Queue()
        pcm_queue: thread_queue.Queue = thread_queue.Queue()
        loop = asyncio.get_running_loop()

        def _blocking_task() -> None:
            try:
                recognition_config = cloud_speech.RecognitionConfig(
                    explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                        encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=_SAMPLE_RATE,
                        audio_channel_count=1,
                    ),
                    model=self._model,
                    language_codes=["en-US"],
                )
                streaming_config = cloud_speech.StreamingRecognitionConfig(
                    config=recognition_config,
                    streaming_features=cloud_speech.StreamingRecognitionFeatures(
                        interim_results=True,
                    ),
                )

                session_start = time.monotonic()
                SESSION_MAX_SECONDS = 240  # 4-minute hard cap, Google limit is 5

                def request_generator():
                    yield cloud_speech.StreamingRecognizeRequest(
                        recognizer=self._recognizer_path,
                        streaming_config=streaming_config,
                    )

                    # Drain stale chunks — discard anything already queued beyond 500ms worth
                    # These accumulated while the previous session was tearing down
                    MAX_QUEUE_DEPTH = 5  # 5 chunks × 100ms = 500ms max acceptable backlog
                    drained = 0
                    while True:
                        try:
                            stale = pcm_queue.get_nowait()
                            if stale is None:
                                # Got shutdown sentinel — put it back and exit
                                pcm_queue.put(None)
                                return
                            drained += 1
                            if pcm_queue.qsize() <= MAX_QUEUE_DEPTH:
                                # Queue is now shallow enough — stop draining
                                # Put this last chunk back, it's still fresh enough
                                pcm_queue.put(stale)
                                break
                        except thread_queue.Empty:
                            break

                    if drained > 0:
                        logger.debug("GENERATOR: drained %d stale chunks from pcm_queue", drained)

                    chunks_sent = 0
                    pace_start = time.monotonic()  # when we sent the first audio chunk
                    first_chunk_received = False

                    while True:
                        if time.monotonic() - session_start > SESSION_MAX_SECONDS:
                            logger.info("Chirp 2 session at 4-min cap, closing gracefully.")
                            break
                        try:
                            # Blocking get with timeout in the thread
                            chunk = pcm_queue.get(timeout=0.1)
                            if chunk is None:
                                break

                            if not first_chunk_received:
                                # Reset pace clock on first real chunk
                                pace_start = time.monotonic()
                                first_chunk_received = True

                            # Pace: chunk N should be sent at N * 100ms from pace_start
                            expected_send_time = chunks_sent * 0.100
                            elapsed = time.monotonic() - pace_start
                            sleep_needed = expected_send_time - elapsed
                            if sleep_needed > 0.005:  # only sleep if meaningfully ahead (>5ms)
                                time.sleep(sleep_needed)

                            logger.debug("GENERATOR: got chunk from pcm_queue, sending to Chirp2")
                            yield cloud_speech.StreamingRecognizeRequest(
                                audio=float32_to_pcm16(chunk)
                            )
                            chunks_sent += 1

                        except thread_queue.Empty:
                            continue

                responses = self._client.streaming_recognize(requests=request_generator())
                logger.debug("CHIRP2: response iterator obtained, entering response loop")
                for response in responses:
                    logger.debug("CHIRP2: received response object, processing...")
                    interim_parts: list[str] = []

                    for result in response.results:
                        if not result.alternatives:
                            continue
                        text = result.alternatives[0].transcript.strip()
                        if not text:
                            continue
                        if result.is_final:
                            if interim_parts:
                                combined = " ".join(interim_parts)
                                asyncio.run_coroutine_threadsafe(
                                    output_queue.put(f"[interim]{combined}"), loop
                                )
                                interim_parts = []
                            logger.debug("Chirp 2 yielded: final=True, text=%r", text)
                            asyncio.run_coroutine_threadsafe(output_queue.put(text), loop)
                        else:
                            interim_parts.append(text)

                    if interim_parts:
                        combined = " ".join(interim_parts)
                        logger.debug("Chirp 2 yielded: final=False, combined=%r", combined)
                        asyncio.run_coroutine_threadsafe(
                            output_queue.put(f"[interim]{combined}"), loop
                        )

            except Exception as e:
                logger.error("Chirp 2 blocking session error: %s", e)
            finally:
                asyncio.run_coroutine_threadsafe(output_queue.put(None), loop)

        async def _forwarder() -> None:
            """Drain asyncio audio_queue → thread-safe pcm_queue."""
            try:
                while True:
                    chunk = await audio_queue.get()
                    pcm_queue.put(chunk)
                    if chunk is None:
                        break
            except asyncio.CancelledError:
                pcm_queue.put(None)  # unblock the thread if it's waiting

        # Start both concurrently
        thread_task = asyncio.create_task(asyncio.to_thread(_blocking_task))
        forwarder_task = asyncio.create_task(_forwarder())

        try:
            while True:
                token = await output_queue.get()
                if token is None:
                    break
                yield token
        finally:
            forwarder_task.cancel()
            # We don't unblock _blocking_task here because forwarder handles chunk=None
            await thread_task


if __name__ == "__main__":
    import os
    import wave
    import yaml
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(level=logging.DEBUG)

    config_path = "config.yaml"
    if not os.path.exists(config_path):
        config_path = "../config.yaml"

    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    transcriber = Chirp2Transcriber(full_config["chirp2"])
    test_file = "debug/capture_test.wav"
    if not os.path.exists(test_file):
        test_file = "../debug/capture_test.wav"
    
    if os.path.exists(test_file):
        with wave.open(test_file, "r") as wf:
            raw = wf.readframes(wf.getnframes())
            audio_int16 = np.frombuffer(raw, dtype=np.int16)
            audio_f32 = audio_int16.astype(np.float32) / 32768.0

        async def run_test():
            q = asyncio.Queue()
            async def feed():
                chunk_size = 1600
                for i in range(0, len(audio_f32), chunk_size):
                    await q.put(audio_f32[i:i+chunk_size])
                    await asyncio.sleep(0.1)
                await q.put(None)
            
            async def consume():
                async for res in transcriber.transcribe_stream(q):
                    print(f"RESULT: {res}")

            await asyncio.gather(feed(), consume())

        asyncio.run(run_test())
