"""
Streaming transcription engine — Ephemeral, VAD-driven architecture.
"""

import asyncio
import logging
from collections.abc import Callable, Awaitable

import numpy as np

from transcription.base import AbstractTranscriber
from utils.transcript_writer import TranscriptWriter

logger = logging.getLogger(__name__)


class StreamingTranscriber:
    """
    Engine that manages ephemeral transcription sessions.
    
    Whenever VAD detects speech, a new gRPC session is opened.
    When VAD detects silence, the session is closed gracefully.
    """

    def __init__(
        self,
        transcriber: AbstractTranscriber,
        config: dict,
        transcript_writer: TranscriptWriter | None = None,
    ) -> None:
        self._transcriber = transcriber
        self._config = config
        self._transcript_writer = transcript_writer
        
        # Event to signal utterance end (silence)
        self._utterance_end_event = asyncio.Event()

    def notify_utterance_end(self) -> None:
        """Signal from VAD that speech has stopped."""
        self._utterance_end_event.set()

    async def run(
        self,
        audio_queue: asyncio.Queue,
        text_callback: Callable[[str], Awaitable[None]],
        newline_callback: Callable[[], Awaitable[None]],
        interim_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        """
        Main execution loop.
        """
        logger.info("VAD-driven streaming engine active.")
        
        while True:
            # 1. Wait for the very first chunk of speech to arrive
            first_chunk = await audio_queue.get()
            if first_chunk is None: break # Shutdown sentinel
            
            logger.debug("Speech detected! Opening ephemeral transcription session...")
            
            # 2. Open an ephemeral session
            session_queue = asyncio.Queue()
            await session_queue.put(first_chunk)
            
            # 3. Define the task to process this session's responses
            async def _process_session(q):
                try:
                    async for token in self._transcriber.transcribe_stream(q):
                        if not token: continue
                        
                        if token.startswith("[interim]"):
                            if interim_callback:
                                await interim_callback(token[len("[interim]"):])
                        else:
                            await text_callback(token)
                            if self._transcript_writer:
                                self._transcript_writer.append(token)
                except Exception as e:
                    logger.error("Error in ephemeral session: %s", e)

            session_task = asyncio.create_task(_process_session(session_queue))
            
            # 4. Feed subsequent chunks until utterance end
            self._utterance_end_event.clear()
            
            while not self._utterance_end_event.is_set():
                if session_task.done():
                    logger.debug("Session task finished early (likely 4-min cap), restarting...")
                    # The old session_queue might still have audio? 
                    # Unlikely if the task finished normally.
                    session_queue = asyncio.Queue()
                    session_task = asyncio.create_task(_process_session(session_queue))

                try:
                    # Wait for next chunk OR the utterance end signal OR task completion
                    chunk_task = asyncio.create_task(audio_queue.get())
                    vad_task = asyncio.create_task(self._utterance_end_event.wait())
                    
                    done, pending = await asyncio.wait(
                        [chunk_task, vad_task, session_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    for p in pending: 
                        if p is not session_task:
                            p.cancel()
                    
                    if session_task in done:
                        # Session task finished (e.g. error or cap)
                        try:
                            session_task.result()
                        except Exception as e:
                            logger.error("Session task failed: %s", e)
                        continue

                    if chunk_task in done:
                        chunk = chunk_task.result()
                        if chunk is None:
                            self._utterance_end_event.set()
                            break
                        await session_queue.put(chunk)
                    
                    if vad_task in done:
                        # VAD detected silence
                        break
                        
                except asyncio.CancelledError:
                    break

            # 5. Clean up the session with timeout
            logger.debug("Silence detected. Closing ephemeral session.")
            if not session_task.done():
                await session_queue.put(None)  # signal gRPC stream to close
                try:
                    await asyncio.wait_for(asyncio.shield(session_task), timeout=3.0)
                except asyncio.TimeoutError:
                    logger.debug("Session teardown timed out after 3s — cancelling.")
                    session_task.cancel()
                    try:
                        await session_task
                    except (asyncio.CancelledError, Exception):
                        pass

            # 6. Flush UI and storage
            await newline_callback()
            if self._transcript_writer:
                self._transcript_writer.flush()

            logger.debug("Ephemeral session finalized.")
