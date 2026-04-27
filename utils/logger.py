"""
Structured logging setup for the LiveRex pipeline.

Call setup_logging() once at process start (from main.py) before any other
module logs anything.  All loggers in the project inherit their level from the
root logger configured here.

Noisy third-party loggers (faster_whisper, torch, numba, google) are silenced
to WARNING so they don't flood the terminal during normal operation.  Use
--debug to restore them to DEBUG.
"""

import logging
import sys


def setup_logging(debug: bool = False) -> None:
    """
    Configure logging for the entire process.

    Sets up a single StreamHandler to stderr with a timestamped format.
    Noisy third-party libraries are raised to WARNING regardless of the
    debug flag to keep terminal output readable.

    Args:
        debug: If True, set root level to DEBUG (verbose pipeline tracing).
               If False, set root level to INFO (normal operation).

    Returns:
        None
    """
    level = logging.DEBUG if debug else logging.INFO

    fmt = "%(asctime)s.%(msecs)03d  %(levelname)-8s  %(name)-28s  %(message)s"
    datefmt = "%H:%M:%S"

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if setup_logging() is called more than once
    root.handlers.clear()
    root.addHandler(handler)

    # Third-party loggers that are too verbose even in debug mode
    _noisy = [
        "faster_whisper",
        "numba",
        "torch",
        "google.auth",
        "google.api_core",
        "urllib3",
        "httpcore",
        "httpx",
    ]
    for name in _noisy:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug("Logging configured (level=%s)", logging.getLevelName(level))
