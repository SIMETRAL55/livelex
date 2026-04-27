"""
Async audio capture from PulseAudio/PipeWire monitor sources.

Opens a sounddevice InputStream pointed at a monitor source (which captures
all system audio output — including Google Meet). Each callback converts the
raw hardware audio to pipeline format (16kHz mono float32) and pushes it into
an asyncio.Queue for downstream consumers.

The sounddevice callback is non-blocking: it does no I/O, no sleeping, and
no heavy computation. Resampling via audio_utils is the only work done there
and is bounded to a small fixed-size numpy array.
"""

import asyncio
import logging
import subprocess

import numpy as np
import sounddevice as sd

from utils import audio_utils

logger = logging.getLogger(__name__)


class AudioCapture:
    """
    Captures system audio from a PulseAudio/PipeWire monitor source.

    Usage::

        capture = AudioCapture()
        queue: asyncio.Queue = asyncio.Queue()
        await capture.start(queue)
        # ... consume chunks from queue ...
        await capture.stop()
    """

    def __init__(
        self,
        device: str | None = None,
        sample_rate: int = 16000,
        chunk_ms: int = 100,
    ):
        """
        Args:
            device: sounddevice device name string. If None, auto-detect monitor source
                    via PulseAudio/PipeWire (recommended — device names can change).
            sample_rate: Target output sample rate in Hz. Chunks in the queue will be
                         at this rate. (default 16000)
            chunk_ms: Size of each audio chunk in milliseconds. Each item in the queue
                      will contain approximately sample_rate * chunk_ms / 1000 samples.
                      (default 100)
        """
        self._device = device
        self._sample_rate = sample_rate
        self._chunk_ms = chunk_ms
        self._stream: sd.InputStream | None = None
        self._resolved_device: int | str | None = None

    def _resolve_device(self) -> str:
        """
        Determine which audio device to open.

        Resolution order:
          1. Explicit device arg passed to __init__
          2. Default monitor source from `pactl info`
          3. First monitor source from `pactl list sources`

        Args:
            (none)

        Returns:
            Device name string suitable for sounddevice.

        Raises:
            RuntimeError: If no monitor source can be found. The error message
                          includes the output of `pactl list sources short` so
                          the user can see what IS available.
        """
        if self._device is not None:
            logger.debug("Using explicitly configured device: %s", self._device)
            return self._device

        default = audio_utils.get_default_monitor_source()
        if default is not None:
            logger.debug("Auto-detected default monitor source: %s", default)
            return default

        monitors = audio_utils.list_monitor_sources()
        if monitors:
            name = monitors[0]["name"]
            logger.debug("Using first available monitor source: %s", name)
            return name

        # No monitor source found — show user what sources exist before crashing
        try:
            result = subprocess.run(
                ["pactl", "list", "sources", "short"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            sources_output = result.stdout or "(empty output)"
        except Exception as exc:
            sources_output = f"(could not run pactl: {exc})"

        raise RuntimeError(
            "No PulseAudio/PipeWire monitor source found.\n\n"
            "Available sources (from `pactl list sources short`):\n"
            f"{sources_output}\n"
            "A monitor source captures system audio output. It usually ends in '.monitor'.\n"
            "Ensure PulseAudio or PipeWire is running and an audio output device is active."
        )

    async def start(self, queue: asyncio.Queue) -> None:
        """
        Start capturing audio and feeding chunks into queue.

        Opens a sounddevice InputStream at the hardware's native sample rate,
        then resamples each chunk to the target rate inside the callback.
        Chunks placed in the queue are always: 16kHz, mono, float32, [-1.0, 1.0].

        The sounddevice callback runs in a PortAudio background thread. It uses
        loop.call_soon_threadsafe to enqueue without blocking the audio thread.

        Args:
            queue: asyncio.Queue that will receive float32 numpy arrays.
                   Each array has shape (N,) where N ≈ sample_rate * chunk_ms / 1000.

        Raises:
            RuntimeError: If no monitor source is found, or if neither 'pipewire' nor
                          'default' device appears in sounddevice's device list.
            sounddevice.PortAudioError: If the device cannot be opened.
        """
        # ── Device resolution ────────────────────────────────────────────────
        # On PipeWire, monitor sources are not separate PortAudio devices.
        # The correct approach is:
        #   1. Tell PipeWire which source to use  (pactl set-default-source)
        #   2. Open the 'pipewire' (or 'default') virtual PortAudio device
        #      — PipeWire will route it to whatever source we set in step 1.
        #
        # Exception: if the caller passed an explicit non-monitor device name,
        # we trust them and skip pactl setup entirely.

        pactl_source_name: str | None = None  # set when we need to configure PipeWire

        if self._device is not None:
            if self._device.endswith(".monitor"):
                # Explicit pactl monitor name — still need to set PipeWire default
                pactl_source_name = self._device
            else:
                # Explicit sounddevice name (e.g. "hw:0,0") — use as-is, skip pactl
                logger.debug("Using explicit sounddevice device name: %r", self._device)
                self._resolved_device = self._device
        else:
            # Auto-detect: prefer the default sink's monitor, fall back to first monitor
            pactl_source_name = audio_utils.get_default_monitor_source()
            if pactl_source_name is None:
                monitors = audio_utils.list_monitor_sources()
                if monitors:
                    pactl_source_name = monitors[0]["name"]
                    logger.debug("Using first available monitor source: %s", pactl_source_name)
                else:
                    try:
                        result = subprocess.run(
                            ["pactl", "list", "sources", "short"],
                            capture_output=True, text=True, timeout=5,
                        )
                        sources_output = result.stdout or "(empty output)"
                    except Exception as exc:
                        sources_output = f"(could not run pactl: {exc})"
                    raise RuntimeError(
                        "No PulseAudio/PipeWire monitor source found.\n\n"
                        "Available sources (from `pactl list sources short`):\n"
                        f"{sources_output}\n"
                        "Ensure PulseAudio or PipeWire is running and an audio output device is active."
                    )

        if pactl_source_name is not None:
            ok = audio_utils.set_default_pipewire_source(pactl_source_name)
            if ok:
                logger.info("Set PipeWire default source to: %s", pactl_source_name)
            else:
                logger.warning(
                    "pactl set-default-source failed for %r — stream may capture wrong source",
                    pactl_source_name,
                )

            device_index = audio_utils.get_pipewire_device_index()
            if device_index is None:
                all_devs = sd.query_devices()
                dev_list = "\n".join(
                    f"  [{i:>2}] {d['name']}"
                    f"  (in={d['max_input_channels']} out={d['max_output_channels']})"
                    for i, d in enumerate(all_devs)
                )
                logger.error(
                    "Neither 'pipewire' nor 'default' input device found.\n"
                    "All PortAudio devices:\n%s", dev_list,
                )
                raise RuntimeError(
                    "Could not find 'pipewire' or 'default' device in sounddevice.\n\n"
                    f"All PortAudio devices:\n{dev_list}\n\n"
                    "Ensure PipeWire is running and pipewire-pulse is installed:\n"
                    "  sudo apt install pipewire-pulse"
                )
            self._resolved_device = device_index

        # After resolution, self._resolved_device is either an int (PipeWire virtual
        # device index) or a str (explicit raw device name). Both work with sounddevice.
        dev = self._resolved_device

        # Query native sample rate so we know how much to resample
        try:
            dev_info = sd.query_devices(dev, kind="input")
            native_rate = int(dev_info["default_samplerate"])
        except Exception:
            logger.warning(
                "Could not query device %r — opening at target rate %dHz",
                dev,
                self._sample_rate,
            )
            native_rate = self._sample_rate

        native_chunk_frames = int(native_rate * self._chunk_ms / 1000)
        loop = asyncio.get_running_loop()

        def _callback(
            indata: np.ndarray,
            frames: int,
            time_info,
            status: sd.CallbackFlags,
        ) -> None:
            """
            PortAudio callback — must never block.

            Copies the indata buffer (a view that PortAudio will recycle),
            converts it to pipeline format, and schedules a queue.put_nowait
            on the asyncio event loop.

            Args:
                indata: Raw audio from PortAudio. Shape (frames, channels). float32.
                frames: Number of audio frames in indata.
                time_info: PortAudio timing struct (unused).
                status: Callback flags (underflow/overflow indicators).
            """
            if status:
                logger.warning("sounddevice callback status: %s", status)

            # indata is a view into a recycled buffer — copy before it's overwritten
            chunk = audio_utils.convert_to_pipeline_format(indata.copy(), native_rate)
            loop.call_soon_threadsafe(queue.put_nowait, chunk)

        logger.info(
            "Opening audio stream: device=%r  native=%dHz  chunk=%dms  target=%dHz",
            dev,
            native_rate,
            self._chunk_ms,
            self._sample_rate,
        )

        self._stream = sd.InputStream(
            device=dev,
            samplerate=native_rate,
            blocksize=native_chunk_frames,
            channels=2,
            dtype="float32",
            callback=_callback,
        )
        self._stream.start()
        logger.info("Audio capture started (device=%r)", dev)

    async def stop(self) -> None:
        """
        Stop the audio stream cleanly.

        Safe to call even if start() was never called or if already stopped.

        Args:
            (none)

        Returns:
            None
        """
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio capture stopped")

    def get_device_info(self) -> dict:
        """
        Return sounddevice metadata for the active input device.

        Args:
            (none)

        Returns:
            Dict from sounddevice.query_devices(). Useful keys:
              - name: str
              - default_samplerate: float
              - max_input_channels: int
              - hostapi: int

        Raises:
            RuntimeError: If called before start() (device not yet resolved).
        """
        if self._resolved_device is None:
            raise RuntimeError(
                "get_device_info() called before start(). "
                "Call start() first so the device can be resolved."
            )
        return dict(sd.query_devices(self._resolved_device, kind="input"))


# ---------------------------------------------------------------------------
# __main__ test block — exempt from no-print rule (CLAUDE.md §Hard Rules)
# Run with: python -m audio.capture --test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os
    import time
    import wave

    parser = argparse.ArgumentParser(
        description="LiveRex audio capture test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Captures system audio and saves it to debug/capture_test.wav.\n"
            "Join a Google Meet call and speak while this is running to verify\n"
            "that Meet audio is captured correctly."
        ),
    )
    parser.add_argument("--test", action="store_true", help="Run capture test (required)")
    parser.add_argument(
        "--duration", type=int, default=10, metavar="SECONDS",
        help="How many seconds to capture (default: 10)"
    )
    args = parser.parse_args()

    if not args.test:
        parser.print_help()
        raise SystemExit(1)

    # Step 1: list all available monitor sources
    print("=" * 60)
    print("LiveRex — Audio Capture Test")
    print("=" * 60)
    print()
    print("Available monitor sources:")
    monitors = audio_utils.list_monitor_sources()
    if monitors:
        for m in monitors:
            print(f"  [{m['index']:>3}]  {m['name']}")
            if m.get("description"):
                print(f"         {m['description']}")
    else:
        print("  (none found — check that PulseAudio/PipeWire is running)")
        print()
        print("Try: pactl list sources short")
        raise SystemExit(1)

    # Step 2: configure PipeWire and show which device will be used
    print()
    chosen_source = audio_utils.get_default_monitor_source() or (monitors[0]["name"] if monitors else None)
    if chosen_source:
        print(f"Setting PipeWire default source to: {chosen_source}")
        audio_utils.set_default_pipewire_source(chosen_source)
        sd_index = audio_utils.get_pipewire_device_index()
        if sd_index is not None:
            dev_name = sd.query_devices(sd_index)["name"]
            print(f"Using sounddevice device index: {sd_index} ({dev_name})")
        else:
            print("WARNING: could not find 'pipewire' or 'default' device in sounddevice")
            print("Ensure PipeWire is running and pipewire-pulse is installed:")
            print("  sudo apt install pipewire-pulse")
            raise SystemExit(1)
    else:
        print("ERROR: no monitor source available")
        raise SystemExit(1)

    print()
    print(f"Capturing {args.duration} seconds of audio...")
    print("Join a Google Meet call and speak to verify audio is flowing.")
    print()

    os.makedirs("debug", exist_ok=True)

    async def _run_test() -> None:
        queue: asyncio.Queue = asyncio.Queue()
        capture = AudioCapture()

        await capture.start(queue)

        info = capture.get_device_info()
        print(
            f"Device opened: {info['name']}  "
            f"({int(info['default_samplerate'])}Hz, "
            f"{info['max_input_channels']}ch)"
        )
        print()

        all_chunks: list[np.ndarray] = []
        start_time = time.perf_counter()
        last_meter_time = start_time

        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= args.duration:
                break

            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue

            all_chunks.append(chunk)

            # Print RMS level meter once per second
            now = time.perf_counter()
            if now - last_meter_time >= 1.0:
                level = audio_utils.rms(chunk)
                bar_len = int(level * 40)
                bar = "#" * bar_len + "-" * (40 - bar_len)
                print(f"  t={elapsed:5.1f}s  RMS={level:.4f}  |{bar}|")
                last_meter_time = now

        await capture.stop()

        if not all_chunks:
            print("\nNo audio captured — is the monitor source working?")
            return

        audio_data = np.concatenate(all_chunks)
        duration_actual = len(audio_data) / 16000
        print(f"\nCaptured {len(audio_data):,} samples ({duration_actual:.2f}s at 16kHz)")

        # Save as 16-bit WAV
        output_path = "debug/capture_test.wav"
        audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        with wave.open(output_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())

        print(f"\nSaved to {output_path} — play it back to verify Meet audio was captured")
        print(f"  Playback: aplay {output_path}")

    asyncio.run(_run_test())
