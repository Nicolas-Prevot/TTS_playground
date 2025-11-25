# F5-TTS Adapter

This directory contains the adapter for **F5-TTS**, a cutting-edge non-autoregressive text-to-speech system. It utilizes **Flow Matching** with a Diffusion Transformer (DiT) backbone to generate speech by refining noise, offering an excellent balance of high inference speed and robust, natural zero-shot voice cloning.


## 🧠 Model Overview

| Feature | Details |
| :--- | :--- |
| **Architecture** | Diffusion Transformer (DiT) with ConvNeXt text encoder. Trained via Flow Matching. |
| **Model Size** | **~335M Parameters** (Base version). |
| **Languages** | **English & Chinese** (Mandarin). Excellent at code-switching between the two. |
| **Voice Cloning** | **Yes (Zero-shot)**. Requires ~10–15s of reference audio **AND** its transcript. |
| **Emotion Control** | **Implicit**. No explicit tags; emotion is inherited from the reference audio style. |
| **Vocoder** | Supports **Vocos** (default, faster) and **BigVGAN** (higher fidelity). |
| **Inference Speed** | Very Fast (Non-autoregressive). Real-Time Factor ~0.15x on consumer GPUs. |

-----

## ⚙️ Installation

This adapter is designed to run as a standalone isolated environment within the `TTS_playground`.

### Prerequisites

  * **Python 3.12** (managed by `uv`).
  * **GPU Recommended:** A generic GPU (CUDA) is highly recommended for diffusion steps.

### Setup via `uv`

Navigate to the adapter directory and sync dependencies:

```bash
cd adapters/f5tts
uv sync
````

*This installs `f5-tts`, `torch`, `torchaudio`, and `vocos`/`bigvgan` specifically for this adapter.*

-----

## 💻 Usage: Local Python

You can use the adapter directly in Python scripts. Unlike some other models, F5-TTS **requires the transcript** of the reference audio to perform accurate cloning.

### Basic Example

*(Adapted from `examples/run_local.py`)*

```python
from tts_adapter_f5tts.adapter import F5TTSAdapter

# 1. Initialize & Load
# model_name options: "F5TTS_v1_Base", "E2TTS_Base"
tts = F5TTSAdapter(
    model_name="F5TTS_v1_Base",
    vocoder_name="vocos", 
    device="cuda"
)
tts.load_model()

# 2. Clone a Voice (Audio + Transcript is MANDATORY)
# Reference audio should be ~15 seconds of clean speech
tts.clone_voice(
    ref_audio="data/ref/basic_ref_en.wav",
    ref_text="Some call me nature, others call me mother nature."
)

# 3. Synthesize (Standard)
audio_bytes = tts.synthesize(
    "F5-TTS uses a diffusion transformer to generate speech.",
    speed=1.0,
    nfe_step=32
)

# 4. Synthesize (Fast / Low Latency)
fast_bytes = tts.synthesize(
    "Generating this with fewer diffusion steps for speed.",
    nfe_step=16, # Fewer steps = faster but potential metallic artifacts
    speed=1.1
)

# Save to disk
with open("output_f5tts.wav", "wb") as f:
    f.write(audio_bytes)
```

### Synthesis Parameters

The `synthesize` method exposes specific diffusion and flow-matching parameters:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `speed` | `1.0` | Speaking rate. `>1.0` is faster, `<1.0` is slower. |
| `nfe_step` | `32` | **Number of Function Evaluations.** The number of denoising steps. `16` is fast (lower quality), `64` is high fidelity (slower). |
| `cfg_strength` | `2.0` | **Classifier-Free Guidance.** Controls how strictly the model adheres to the text. Higher values are stricter, lower values are more "relaxed." |
| `sway_sampling_coef` | `-1.0` | Controls the sampling trajectory. Small negative values usually improve stability. |
| `cross_fade_duration` | `0.2` | Duration (in seconds) of the cross-fade overlap when stitching long text chunks. |

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
    adapter="f5tts",
    clone_voice={ 
        "ref_audio": ref_blob,
        "ref_text": "Some call me nature, others call me mother nature." # Required!
    },
    synthesize={
        "text": "Hello via API.",
        "kwargs": { "nfe_step": 32, "speed": 1.0 }
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
  "adapter": "f5tts",
  "init": { 
    "model_name": "F5TTS_v1_Base", 
    "vocoder_name": "vocos",
    "device": "cuda" 
  },
  "load_model": {},
  "clone_voice": {
    "ref_audio": {
      "name": "ref.wav",
      "b64": "<base64_encoded_wav_bytes>"
    },
    "ref_text": "Transcript of the reference audio is required here."
  },
  "synthesize": {
    "text": "This is a raw HTTP request test.",
    "kwargs": {
      "nfe_step": 32,
      "cfg_strength": 2.0
    }
  }
}
```

-----

## 🎙️ Voice Cloning Best Practices

1.  **Transcript is Key:** Unlike Chatterbox, F5-TTS *requires* the text transcript of the reference audio (`ref_text`) to align the diffusion process. Inaccurate transcripts leads to poor cloning.
2.  **Reference Length:** Optimal length is **10 to 20 seconds**. Extremely short clips (\<5s) may not provide enough timbre data; extremely long clips (\>45s) can sometimes degrade performance.
3.  **Emotion Mirroring:** F5-TTS does not use emotion tags (e.g., `[sad]`). To generate sad speech, use a reference audio clip where the speaker sounds sad. The model mirrors the style of the input.
4.  **Vocoder Choice:** Use `vocos` for speed (default). Use `bigvgan` if you notice artifacts in the high-frequencies, though it consumes more VRAM.

## 🔗 Credits & License

  * **Original Model:** [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)
  * **License:** The F5-TTS code and weights are released under the **MIT License**.
