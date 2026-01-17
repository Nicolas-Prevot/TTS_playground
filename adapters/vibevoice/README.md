# VibeVoice Adapter

This directory contains the adapter for **VibeVoice**, a LLaMA-based TTS model that uses a Diffusion Head (DDPM) for acoustic refinement. It natively supports multi-speaker generation within a single script using speaker labels.

## 🧠 Model Overview

| Feature | Details |
| :--- | :--- |
| **Architecture** | LLaMA Backbone + Diffusion Head (DDPM). |
| **Model Size** | **1.5B** or **7B** variants. |
| **Languages** | Primarily English. |
| **Voice Cloning** | **Yes (Prefill)**. Conditions the generation on reference audio tokens. |
| **Multi-Speaker** | **Yes**. Supports `Speaker N:` text syntax to switch voices dynamically. |
| **Inference** | Requires GPU (CUDA). Uses Flash Attention 2 (with SDPA fallback). |

-----

## ⚙️ Installation

```bash
cd adapters/vibevoice
uv sync
````

-----

## 💻 Usage: Local Python

VibeVoice is unique in how it handles multi-speaker parsing directly from the input text string.

### Basic Example

```python
from tts_adapter_vibevoice.adapter import VibeVoiceAdapter

# 1. Initialize
tts = VibeVoiceAdapter(
    model_id="vibevoice/VibeVoice-7B",
    device="cuda",
    cfg_scale=1.3
)
tts.load_model()

# 2. Clone Voice (Single Speaker)
tts.clone_voice("data/ref/basic_ref_en.wav")

audio = tts.synthesize("Hello, I am now cloning this voice.")

# 3. Multi-Speaker
# Define a mapping for speakers 1 and 2
speaker_map = {
    "1": "data/ref/speaker_A.wav",
    "2": "data/ref/speaker_B.wav"
}
tts.clone_voice(speaker_voices=speaker_map)

# Use syntax "Speaker N:" in text
dialogue = """
Speaker 1: Hey, how are you?
Speaker 2: I'm doing great, thanks for asking!
"""
audio_dialogue = tts.synthesize(dialogue)
```

### Synthesis Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `cfg_scale` | `1.3` | Classifier-Free Guidance scale. Higher = stricter adherence to text/style. |
| `ddpm_steps` | `10` | Number of diffusion steps for the acoustic head. |
| `is_prefill` | `True` | If `True`, uses the reference audio for voice cloning. If `False`, uses the model's internal voice. |
| `generation_config` | `{"do_sample": False}` | Standard Hugging Face generation parameters (temp, top\_p, etc.). |

-----

## 🌐 Usage: API

**POST** `http://localhost:7000/v1/tts`

```json
{
  "adapter": "vibevoicetts",
  "init": { "model_id": "vibevoice/VibeVoice-7B", "device": "cuda" },
  "load_model": {},
  "clone_voice": {
    "ref_audio": { "name": "ref.wav", "b64": "..." }
  },
  "synthesize": {
    "text": "Speaker 1: Hello!",
    "kwargs": { "cfg_scale": 1.5 }
  }
}
```