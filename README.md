# LiveLex — Real-Time Meeting Transcription

A high-fidelity real-time speech transcription tool for Ubuntu that captures audio from any browser or application (Google Meet, Zoom, etc.) and displays live captions in a floating glass overlay.

![LiveLex Demo](https://raw.githubusercontent.com/SIMETRAL55/LiveLex/main/img/demo.gif)

**Features:**
- **Massive Glass UI** — Always-on-top, transparent floating display with modern dark aesthetic.
- **Dual Backends** — Local (Whisper) for privacy or Google Chirp 2 for ultra-low latency.
- **Stable Streaming** — Uses LocalAgreement-2 and ephemeral gRPC streams for flicker-free updates.
- **Pill Mode** — Collapsible minimal HUD for non-intrusive monitoring.

---

## Requirements

- Ubuntu 20.04+ (PipeWire or PulseAudio)
- Python 3.11+
- NVIDIA GPU with CUDA (optional but recommended for local mode)
- Google Cloud Project with Speech-to-Text V2 API enabled (for Chirp 2)

---

## Setup

### 1. System dependencies

```bash
sudo apt install portaudio19-dev python3-dev ffmpeg pipewire-pulse libxcb-cursor0
```

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
 
### 3. Authentication (Chirp 2 only)

LiveLex uses Application Default Credentials (ADC) for Google Cloud Speech-to-Text V2.

1. **GCP Auth**: Set up Application Default Credentials:
   ```bash
   gcloud auth application-default login
   ```
2. **Environment**: Create a `.env` file in the root if you need specific project overrides:
   ```bash
   GCP_PROJECT_ID=your-project-id
   ```

---

## Configuration

Edit `config.yaml` to tune the pipeline. Key sections:

### Transcription
| Key | Default | Description |
|-----|---------|-------------|
| `transcription.backend` | `local` | `local` or `chirp2` |
| `transcription.language` | `en` | BCP-47 code: `en`, `ja`, `ko`, `zh` |

### UI & Audio
| Key | Default | Description |
|-----|---------|-------------|
| `overlay.opacity` | `0.96` | Window transparency (0-1) |
| `vad.threshold` | `0.5` | Speech probability cutoff (0–1) |

---

## Running

```bash
# Local model, English (default)
python main.py

# Chirp 2 backend, Japanese
python main.py --backend chirp2 --language ja

# Save session transcript to a file
python main.py --save-transcript

# Debug logging
python main.py --debug
```

Start a meeting call, then start LiveLex. Captions appear in the floating window.

---

## Interactive Controls

### Mouse
- **Move**: Drag the top HUD bar (containing "LIVELEX") to reposition the window.
- **Resize**: Drag edges or corners to resize the layout.
- **Collapse**: Click the `—` button in the HUD to collapse into a minimal "Pill Mode".

### Global Hotkeys
Available even when LiveLex is not focused:
| Hotkey | Action |
|--------|--------|
| **Ctrl+Shift+H** | Toggle overlay visibility (Hide/Show) |

---

## Architecture

```
AudioCapture (PipeWire monitor)
  → audio_queue
  → VADProcessor (Silero VAD)
      → on_utterance_end → StreamingTranscriber
  → StreamingTranscriber (LocalAgreement-2)
      → on_text/on_newline → CaptionOverlay (Transcript Display)
```

See `ARCHITECTURE.md` for full system design.

---

## Troubleshooting

**No audio captured / empty captions**
```bash
# List monitor sources
pactl list sources short | grep monitor
# Ensure PipeWire is running
pactl info | grep "Server Name"
```

**ALSA errors in terminal** — These are harmless. Suppress with:
```bash
PYTHONWARNINGS=ignore python main.py
```

**CUDA out of memory** — Switch to mixed precision in `config.yaml`:
```yaml
local:
  compute_type: "int8_float16"
```

---

## Project structure

```
livelex/
├── main.py                  ← entry point & pipeline wiring
├── config.yaml              ← tunable parameters
├── audio/
│   ├── capture.py           ← system audio capture (sounddevice)
│   └── vad.py               ← speech detection (silero)
├── transcription/
│   ├── local.py             ← faster-whisper backend
│   ├── chirp2.py            ← Google Chirp 2 backend
│   └── streaming.py         ← stabilization & agreement algorithm
├── ui/
│   └── overlay.py           ← PyQt6 Massive Glass UI
└── utils/
    ├── transcript_writer.py ← session logging to .txt
    └── audio_utils.py       ← monitor source discovery
```
