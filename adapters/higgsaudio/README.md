# HiggsAudio Adapter

This directory contains the adapter for **Higgs Audio v2**, a state-of-the-art multimodal generative model by BosonAI. Unlike traditional TTS, Higgs Audio functions as a large Audio-Language Model (ALM), capable of generating highly expressive, context-aware speech with nuanced zero-shot voice cloning.

## 🧠 Model Overview

| Feature | Details |
| :--- | :--- |
| **Architecture** | Audio-Language Model. **Llama-3-3B** text backbone + **Dual-FFN Audio Adapter**. |
| **Model Size** | **~5.8B Parameters** (3.6B LLM + 2.2B Adapter). Heavy VRAM usage. |
| **Languages** | **English & Chinese** (Mandarin). Others (Korean, German, Spanish) are experimental. |
| **Voice Cloning** | **Yes (Zero-shot)**. Requires ~5–10s of reference audio **AND** its transcript. |
| **Emotion Control** | **Prompt-Driven**. Controlled via `scene_prompt` (e.g., "Quiet room", "Happy tone") and text context. |
| **Codec** | Unified discretized audio tokenizer (24kHz, 25fps). |
| **Inference Speed** | Moderate. Autoregressive generation. Requires GPU (CUDA/MPS). |

-----

## ⚙️ Installation

This adapter is designed to run as a standalone isolated environment within the `TTS_playground`.

### Prerequisites

* **Python 3.10+** (managed by `uv`).
* **GPU Required:** Due to the 5.8B parameter count, a GPU with **at least 12GB VRAM** (16GB+ recommended) is required.

### Setup via `uv`

Navigate to the adapter directory and sync dependencies:

```bash
cd adapters/higgsaudio
uv sync
````

*This installs `boson-multimodal`, `torch`, `transformers`, and `descript-audio-codec`.*

-----

## 💻 Usage: Local Python

You can use the adapter directly in Python scripts. Higgs Audio is unique in that it accepts a **Scene Prompt** to describe the acoustic environment or speaking style.

### Basic Example

*(Adapted from `examples/run_local.py`)*

```python
from tts_adapter_higgsaudio.adapter import HiggsAudioAdapter

# 1. Initialize & Load
# use_static_kv_cache=True is highly recommended for inference speed
tts = HiggsAudioAdapter(
    model_path="bosonai/higgs-audio-v2-generation-3B-base",
    device="cuda",
    use_static_kv_cache=True, 
    max_new_tokens=4096
)
tts.load_model()

# 2. Clone a Voice (Audio + Transcript + Scene)
# The scene_prompt helps sets the mood/environment
tts.clone_voice(
    ref_audio="data/ref/basic_ref_en.wav",
    ref_text="Some call me nature, others call me mother nature.",
    scene_prompt="A clear, calm voice speaking in a professional studio."
)

# 3. Synthesize
audio_bytes = tts.synthesize(
    "Higgs Audio treats speech synthesis as a language modeling task.",
    temperature=0.95,
    seed=42
)

# Save to disk
with open("output_higgs.wav", "wb") as f:
    f.write(audio_bytes)
```

### Synthesis Parameters

The `synthesize` method exposes specific sampling and generation parameters:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `temperature` | `1.0` | Controls randomness. Lower (`0.7`) is more stable; Higher (`1.0+`) is more expressive. |
| `top_p` | `0.95` | Nucleus sampling probability. |
| `chunk_method` | `None` | Set to `"word"` for long-form generation. Splits text to avoid context limits. |
| `chunk_max_word_num` | `200` | Max words per chunk if `chunk_method` is enabled. |
| `ras_win_len` | `7` | **Repetition Anti-Stuck:** Window length to detect and prevent audio looping. |

-----

## 🌐 Usage: API (Docker Compose)

The `TTS_playground` orchestrator (FastAPI + Celery) can serve this adapter via HTTP.

### 1\. Start the Stack

From the project root:

```bash
docker compose up -d
```

### 2\. Python Client (`TTSClient`)

```python
from tts_playground.client.tts_client import TTSClient

client = TTSClient("http://localhost:7000")
ref_blob = client.pack_file("data/ref/basic_ref_en.wav")

result = client.synth(
    adapter="higgsaudio",
    # Init params are passed to the Adapter constructor server-side
    init={
        "use_static_kv_cache": True,
        "max_new_tokens": 4096
    },
    clone_voice={ 
        "ref_audio": ref_blob,
        "ref_text": "Some call me nature, others call me mother nature.",
        "scene_prompt": "A warm, clear voice." # Unique to Higgs
    },
    synthesize={
        "text": "Hello via API.",
        "kwargs": { "temperature": 0.95 }
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
  "adapter": "higgsaudio",
  "init": { 
    "use_static_kv_cache": true,
    "device": "cuda" 
  },
  "load_model": {},
  "clone_voice": {
    "ref_audio": {
      "name": "ref.wav",
      "b64": "<base64_encoded_wav_bytes>"
    },
    "ref_text": "Transcript of the reference audio is required.",
    "scene_prompt": "A clear voice speaking in a quiet room."
  },
  "synthesize": {
    "text": "This is a raw HTTP request test.",
    "kwargs": {
      "temperature": 0.95,
      "seed": 42
    }
  }
}
```

-----

## 🎙️ Best Practices & Gotchas

1.  **Scene Prompts:** This is the model's superpower. Instead of generic emotion tags, use the `scene_prompt` to describe the audio.
      * *Good:* "A whisper in a library", "An energetic announcer in a stadium", "A sad voice on a telephone call".
      * *Default:* If omitted, use "A clear voice speaking in a quiet room."
2.  **VRAM Usage:** This is a \~6B parameter model. If you encounter OOM (Out of Memory) errors:
      * Ensure `use_static_kv_cache=True` (init param).
      * Lower `max_new_tokens`.
      * Ensure no other adapters are loaded on the same GPU.
3.  **Reference Text:** Like F5-TTS, Higgs requires the **exact transcript** (`ref_text`) of the reference audio to align the latent space correctly.
4.  **Long Text:** For texts longer than a paragraph, set `chunk_method="word"`. The model is autoregressive and may become unstable or cut off if the context window is exceeded without chunking.

## 🔗 Credits & License

  * **Original Model:** [BosonAI/Higgs-Audio](https://github.com/boson-ai/higgs-audio)
  * **Hugging Face:** [bosonai/higgs-audio-v2-generation-3B-base](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base)
  * **License:** Community License (Check BosonAI repo for commercial restrictions).