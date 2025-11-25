# Chatterbox TTS Adapter

This directory contains the adapter for **Chatterbox**, a state-of-the-art open-source TTS model by **Resemble AI**. It is a 0.5B-parameter autoregressive Transformer (based on LLaMA) designed for high-fidelity, zero-shot voice cloning with fine-grained control over emotion and pacing.

## 🧠 Model Overview

| Feature | Details |
| :--- | :--- |
| **Architecture** | Decoder-only Transformer (LLaMA-0.5B backbone) + HiFT-GAN Vocoder. |
| **Model Size** | **\~500M Parameters**. Optimized for sub-200ms latency on GPU. |
| **Languages** | **English (en)** officially supported in this adapter. *(Model family supports 23+ languages).* |
| **Voice Cloning** | **Yes (Zero-shot)**. Requires 3–10 seconds of reference audio. |
| **Emotion Control** | **Yes**. Continuous control via `exaggeration` (intensity) and `cfg_weight` (stability). |
| **Audio Quality** | 24 kHz sample rate. |
| **Safety** | Outputs are tagged with Resemble’s **PerTh neural watermark** for content provenance. |

-----

## ⚙️ Installation

This adapter is designed to run as a standalone isolated environment within the `TTS_playground`.

### Prerequisites

  * **Python 3.12** (managed by `uv`).
  * **GPU Recommended:** A generic GPU (CUDA).

### Setup via `uv`

Navigate to the adapter directory and sync dependencies:

```bash
cd adapters/chatterbox
uv sync
```

*This installs the specific dependencies (`chatterbox-tts`, `torch`, etc.) required for this model without conflicting with other adapters.*

-----

## 💻 Usage: Local Python

You can use the adapter directly in Python scripts to synthesize audio.

### Basic Example

*(Adapted from `examples/run_local.py`)*

```python
from pathlib import Path
from tts_adapter_chatterbox.adapter import ChatterboxTTSAdapter

# 1. Initialize & Load
# device defaults to "cuda" or "mps" if available, else "cpu"
tts = ChatterboxTTSAdapter()
tts.load_model()

# 2. Clone a Voice
# Provide a path to a 3-10 second clean WAV file
ref_audio_path = "data/ref/basic_ref_en.wav"
tts.clone_voice(ref_audio_path)

# 3. Synthesize (Neutral)
audio_bytes = tts.synthesize(
    "I've been a silent spectator, watching empires rise and fall.",
    cfg_weight=0.5,
    exaggeration=0.5,  # ~Neutral
    temperature=0.8
)

# 4. Synthesize (Dramatic/Emotional)
emotional_bytes = tts.synthesize(
    "But always remember, I am mighty and enduring!",
    cfg_weight=0.4,    # Lower CFG = more relaxed pacing/higher expressiveness
    exaggeration=0.85, # High emotion intensity
    temperature=0.9
)

# Save to disk
with open("output_chatterbox.wav", "wb") as f:
    f.write(emotional_bytes)
```

### Synthesis Parameters

The `synthesize` method exposes several controls to fine-tune the output:

| Parameter | Default | Range | Description |
| :--- | :--- | :--- | :--- |
| `exaggeration` | `0.5` | `0.2`–`1.0` | **Emotion Knob.** `0.5` is neutral. Higher values make speech more dramatic/expressive. |
| `cfg_weight` | `0.5` | `0.2`–`0.8` | **Stability vs. Style.** `0.5` is standard. Lower values (`0.3`) allow more style transfer/slower pacing. `0.0` is used for cross-lingual transfer. |
| `temperature` | `0.8` | `0.6`–`1.0` | Controls randomness. Higher = more varied prosody; Lower = more monotonous/stable. |
| `repetition_penalty` | `1.2` | `1.0`–`1.4` | Penalizes repeating tokens. Increase if the model stutters. |
| `top_p` | `1.0` | `0.8`–`1.0` | Nucleus sampling probability. |

-----

## 🌐 Usage: API (Docker Compose)

The `TTS_playground` orchestrator (FastAPI + Celery) can serve this adapter via HTTP.

### 1\. Start the Stack

From the project root:

```bash
docker compose up -d
```

*This starts the API on port 7000 and the Celery worker managing the adapters.*

### 2\. Python Client (`TTSClient`)

```python
from tts_playground.client.tts_client import TTSClient

client = TTSClient("http://localhost:7000")
ref_blob = client.pack_file("data/ref/basic_ref_en.wav")

result = client.synth(
    adapter="chatterbox",
    clone_voice={ "ref_audio": ref_blob },
    synthesize={
        "text": "Hello via API.",
        "kwargs": { "exaggeration": 0.7, "cfg_weight": 0.5 }
    },
    download=True,
    dest_path="api_output.wav"
)
```

### 3\. Direct HTTP Request

If calling from non-Python environments:

**POST** `http://localhost:7000/v1/tts`

```json
{
  "adapter": "chatterbox",
  "init": { "device": "cuda" },
  "load_model": {},
  "clone_voice": {
    "ref_audio": {
      "name": "ref.wav",
      "b64": "<base64_encoded_wav_bytes>"
    }
  },
  "synthesize": {
    "text": "This is a raw HTTP request test.",
    "kwargs": {
      "exaggeration": 0.6,
      "temperature": 0.8
    }
  }
}
```

-----

## 🎙️ Voice Cloning Best Practices

1.  **Reference Length:** Use **3 to 10 seconds** of audio. Very long clips do not necessarily improve quality and may confuse the style encoder.
2.  **Audio Quality:** Ensure the reference is **clean** (no background music, noise, or reverb). The model will attempt to clone the background noise if present.
3.  **Consistency:** For comparisons, use the same recording conditions.
4.  **State:** In this adapter, calling `clone_voice` caches the audio path/embedding. You can call `synthesize` multiple times without re-cloning.

## 🔗 Credits & License

  * **Original Model:** [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox)
  * **License:** The model weights are released under **CC-BY-NC** (Non-Commercial). Please verify the license on the official repository before commercial use.