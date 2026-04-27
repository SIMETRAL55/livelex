# LiveRex — AI-Powered Meeting Assistant & Real-Time Transcription

An AI-powered meeting assistant for Ubuntu that provides real-time speech transcription and context-aware answers. Captures audio from any browser or application and displays it in a modern, two-panel floating glass UI.

**Features:**
- **AI Meeting Assistant** — Ask questions about the meeting or get automatic answers using Gemini 2.5 Flash.
- **Massive Glass UI** — Always-on-top, transparent dual-panel display (Transcript + AI Answers).
- **Dual Backends** — Local (Whisper) for privacy or Google Chirp 2 for ultra-low latency.
- **Interactive Context** — Click any transcript line to trigger an AI deep-dive.

---

## Requirements

- Ubuntu 20.04+ (PipeWire or PulseAudio)
- Python 3.11+
- NVIDIA GPU with CUDA (optional but recommended for local mode)
- Google Cloud Project with Vertex AI and Speech-to-Text V2 APIs enabled

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
 
### 3. Authentication

LiveRex uses Google Cloud for both Chirp 2 transcription and the AI Assistant (Vertex AI).

1. **GCP Auth**: Set up Application Default Credentials:
   ```bash
   gcloud auth application-default login
   ```
2. **Environment**: Create a `.env` file in the root:
   ```bash
   GCP_PROJECT_ID=your-project-id
   GCP_LOCATION=us-central1
   ```

---

## Configuration

Edit `config.yaml` to tune the pipeline. Key sections:

### Transcription
| Key | Default | Description |
|-----|---------|-------------|
| `transcription.backend` | `local` | `local` or `chirp2` |
| `transcription.language` | `en` | BCP-47 code: `en`, `ja`, `ko`, `zh` |

### AI Assistant (Gemini)
| Key | Default | Description |
|-----|---------|-------------|
| `assistant.enabled` | `true` | Show/hide the AI panel and detection logic |
| `assistant.auto_detect` | `true` | Automatically answer detected questions |
| `assistant.model` | `gemini-2.5-flash` | Model ID used for answering |
| `assistant.system_prompt`| (Helpful assistant) | Instructions for the AI personality |

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

Start a meeting call, then start LiveRex. Captions appear in the floating window.

---

## Interactive Controls

### Mouse
- **Ask AI**: Click any row in the **Transcript Panel** to manually trigger an AI answer for that specific text.
- **Move**: Drag the top HUD bar (containing "LIVEREX") to reposition the window.
- **Resize**: Drag edges or corners to resize the two-panel layout.
- **Collapse**: Click the `—` button in the HUD to collapse into a minimal "Pill Mode".
- **Toggle Answers**: Click the `‹` / `›` buttons to hide/show the AI panel.

### Global Hotkeys
Available even when LiveRex is not focused:
| Hotkey | Action |
|--------|--------|
| **Ctrl+Shift+A** | Force AI to answer the most recent speaker utterance |
| **Ctrl+Shift+H** | Toggle overlay visibility (Hide/Show) |

---

## Architecture

```
AudioCapture (PipeWire monitor)
  → audio_queue
  → VADProcessor (Silero VAD)
      → on_utterance_end → StreamingTranscriber
  → StreamingTranscriber (LocalAgreement-2)
      → on_text/on_newline → CaptionOverlay (Transcript Panel)
      → on_newline → QuestionAnswerer (Gemini 2.5 Flash)
          → add_answer → CaptionOverlay (AI Answers Panel)
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
LiveRex/
├── main.py                  ← entry point & pipeline wiring
├── config.yaml              ← tunable parameters
├── audio/
│   ├── capture.py           ← system audio capture (sounddevice)
│   └── vad.py               ← speech detection (silero)
├── transcription/
│   ├── local.py             ← faster-whisper backend
│   ├── chirp2.py            ← Google Chirp 2 backend
│   ├── answerer.py          ← Gemini AI Assistant logic
│   └── streaming.py         ← stabilization & agreement algorithm
├── ui/
│   └── overlay.py           ← PyQt6 Massive Glass UI
└── utils/
    ├── transcript_writer.py ← session logging to .txt
    └── audio_utils.py       ← monitor source discovery
```
