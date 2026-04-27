"""
Session transcript writer for LiveRex.

Accumulates transcribed text in memory and writes it to a timestamped file
on demand.  Designed to be flushed after each utterance and on shutdown.

File naming convention: ``{session_name}_{YYYYMMDD_HHMMSS}.txt``
The filename is fixed at construction time so all flushes append to the
same file rather than creating new ones during a session.
"""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TranscriptWriter:
    """
    Accumulates session transcript in memory and persists it to disk on demand.

    The output file is named ``{session_name}_{YYYYMMDD_HHMMSS}.txt`` and
    lives in the configured output directory.  The filename is fixed at
    construction time so all flushes go to the same file.

    All writes overwrite the file with the complete transcript so far —
    this ensures the file is always a consistent snapshot even if the
    process exits unexpectedly between flushes.

    Thread safety: not thread-safe.  All calls must originate from the
    asyncio event loop (single-threaded), which is the case in LiveRex.
    """

    def __init__(self, output_dir: str | Path, session_name: str) -> None:
        """
        Configure the writer and resolve the output file path.

        Creates the output directory immediately if it does not already exist.
        The timestamp in the filename is captured at construction time so
        it reflects when the session started, not when the first flush occurs.

        Args:
            output_dir: Directory in which to write transcript files.
                        Created (including parents) if it does not exist.
            session_name: Short human-readable label for the session used
                          as the filename prefix (e.g. ``"session"``).

        Returns:
            None
        """
        self._text: str = ""
        output_path = Path(output_dir)

        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create transcript directory %s: %s", output_path, exc)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._file_path: Path = output_path / f"{session_name}_{timestamp}.txt"

        logger.debug("TranscriptWriter initialised → %s", self._file_path)

    @property
    def file_path(self) -> Path:
        """
        The path of the transcript file that will be written on flush().

        Returns:
            Path object pointing to the output file.
        """
        return self._file_path

    def append(self, text: str) -> None:
        """
        Append a text fragment to the in-memory transcript buffer.

        Args:
            text: Text fragment to append (may be a word, phrase, or partial
                  token as emitted by StreamingTranscriber's text_callback).

        Returns:
            None
        """
        self._text += text

    def flush(self) -> None:
        """
        Write the current in-memory transcript to disk.

        Overwrites the file so it always contains the complete session
        transcript up to this point.  Skips the write when no text has
        been accumulated.  If writing fails, logs a warning and continues —
        never raises, never crashes the pipeline.

        Args:
            (none)

        Returns:
            None
        """
        if not self._text:
            return
        try:
            self._file_path.write_text(self._text, encoding="utf-8")
            logger.debug(
                "Transcript flushed (%d chars) → %s",
                len(self._text),
                self._file_path,
            )
        except OSError as exc:
            logger.warning(
                "Failed to write transcript to %s: %s",
                self._file_path,
                exc,
            )
