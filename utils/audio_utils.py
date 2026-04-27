"""
Audio utility functions for format conversion and device discovery.

All audio entering the pipeline must be: 16kHz, mono, float32, range [-1.0, 1.0].
These functions handle conversion from whatever the hardware gives us.

No classes. No global state. Pure functions only.
"""

import logging
import subprocess
from fractions import Fraction

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

_PIPELINE_RATE = 16000


def convert_to_pipeline_format(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """
    Convert any numpy audio array to pipeline standard: 16kHz mono float32 [-1.0, 1.0].

    Args:
        audio: Input numpy array. Shape (N,) for mono or (N, C) for multi-channel.
               Accepts any integer or float dtype (int16, int32, float32, float64, etc.).
        src_rate: Source sample rate in Hz.

    Returns:
        float32 numpy array, shape (M,), 16kHz, mono, values clipped to [-1.0, 1.0].
    """
    # Normalise integer dtypes to float32 before any other operation
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = max(abs(info.min), info.max)
        audio = audio.astype(np.float32) / scale
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Collapse multi-channel to mono by averaging channels
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError(f"Unsupported audio shape: {audio.shape}. Expected (N,) or (N, C).")

    # Resample to pipeline rate using polyphase filter (avoids quality loss)
    if src_rate != _PIPELINE_RATE:
        frac = Fraction(_PIPELINE_RATE, src_rate).limit_denominator(1000)
        audio = resample_poly(audio, frac.numerator, frac.denominator).astype(np.float32)

    # Hard clip — never let values outside [-1, 1] into the pipeline
    return np.clip(audio, -1.0, 1.0)


def list_monitor_sources() -> list[dict]:
    """
    Query PulseAudio/PipeWire for available monitor sources.

    Parses the verbose output of `pactl list sources` to extract index, name,
    and description for every source whose name ends in ".monitor".

    Args:
        (none)

    Returns:
        List of dicts: [{"name": str, "index": int, "description": str}]
        Returns an empty list if pactl is unavailable or no monitor sources exist.
    """
    try:
        result = subprocess.run(
            ["pactl", "list", "sources"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        logger.warning("pactl not found — PulseAudio/PipeWire may not be running")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("pactl timed out while listing sources")
        return []

    sources: list[dict] = []
    current: dict = {}

    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Source #"):
            # Flush the previous source if it was a monitor
            if current.get("name", "").endswith(".monitor"):
                sources.append(current)
            current = {"index": int(stripped.split("#")[1]), "description": ""}
        elif stripped.startswith("Name:"):
            current["name"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Description:"):
            current["description"] = stripped.split(":", 1)[1].strip()

    # Flush the final source
    if current.get("name", "").endswith(".monitor"):
        sources.append(current)

    logger.debug(
        "Found %d monitor source(s): %s",
        len(sources),
        [s["name"] for s in sources],
    )
    return sources


def get_default_monitor_source() -> str | None:
    """
    Return the name of the default monitor source (the one for the default sink).

    Uses `pactl info` to find the default sink name, then appends ".monitor".
    This is the source that captures all system audio output — what Google Meet
    (and any other app) plays through your speakers.

    Args:
        (none)

    Returns:
        Monitor source name string (e.g. "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"),
        or None if the default sink cannot be determined.
    """
    try:
        result = subprocess.run(
            ["pactl", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        logger.warning("pactl not found — PulseAudio/PipeWire may not be running")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("pactl timed out while querying server info")
        return None

    for line in result.stdout.splitlines():
        if line.startswith("Default Sink:"):
            sink_name = line.split(":", 1)[1].strip()
            monitor = f"{sink_name}.monitor"
            logger.debug("Default monitor source resolved to: %s", monitor)
            return monitor

    logger.warning("Could not find 'Default Sink' in pactl info output")
    return None


def set_default_pipewire_source(pactl_source_name: str) -> bool:
    """
    Set the PipeWire/PulseAudio default input source.

    This must be called before opening a sounddevice stream so PipeWire routes
    the correct source to the virtual 'pipewire'/'default' device. On PipeWire,
    monitor sources are not exposed as separate PortAudio devices — instead,
    setting the default source here causes the 'pipewire' virtual device to
    receive audio from that source.

    Args:
        pactl_source_name: Full pactl source name, e.g.
            "alsa_output.pci-0000_05_00.6.HiFi__hw_Generic_1__sink.monitor"

    Returns:
        True if the command succeeded (returncode == 0), False otherwise.
    """
    try:
        result = subprocess.run(
            ["pactl", "set-default-source", pactl_source_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        logger.warning("pactl not found — cannot set default source")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("pactl timed out while setting default source")
        return False

    if result.returncode != 0:
        logger.warning(
            "pactl set-default-source failed (rc=%d): %s",
            result.returncode,
            result.stderr.strip(),
        )
        return False

    logger.debug("Set PipeWire default source to: %s", pactl_source_name)
    return True


def get_pipewire_device_index() -> int | None:
    """
    Find the sounddevice index for PipeWire's virtual input device.

    On PipeWire, audio routing is controlled via pactl (see set_default_pipewire_source).
    The actual PortAudio device to open is the 'pipewire' virtual device (preferred)
    or 'default' (fallback). Both route to whatever PipeWire's current default source is.

    Args:
        (none)

    Returns:
        int device index of the 'pipewire' input device, or the 'default' input device,
        or None if neither is present in sounddevice's device list.
    """
    all_devices = sd.query_devices()

    pipewire_index: int | None = None
    default_index: int | None = None

    for i, dev in enumerate(all_devices):
        if dev["max_input_channels"] > 0:
            if dev["name"] == "pipewire" and pipewire_index is None:
                pipewire_index = i
            elif dev["name"] == "default" and default_index is None:
                default_index = i

    if pipewire_index is not None:
        logger.debug("get_pipewire_device_index: found 'pipewire' at index %d", pipewire_index)
        return pipewire_index

    if default_index is not None:
        logger.debug("get_pipewire_device_index: 'pipewire' not found, using 'default' at index %d", default_index)
        return default_index

    logger.debug("get_pipewire_device_index: neither 'pipewire' nor 'default' input device found")
    return None


def rms(audio: np.ndarray) -> float:
    """
    Compute the RMS (root mean square) level of an audio array.

    Args:
        audio: numpy array of any shape and dtype.
               Assumed to be in [-1.0, 1.0] range for a meaningful result.

    Returns:
        RMS value as a float. For normalised audio this is in [0.0, 1.0].
        Returns 0.0 for empty arrays.
    """
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """
    Convert a float32 audio array to raw signed-16-bit PCM bytes.

    Args:
        audio: float32 numpy array with values in [-1.0, 1.0].

    Returns:
        Bytes object containing little-endian int16 samples.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()
