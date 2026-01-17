# OpenAudio S1 Mini Adapter

This directory contains the adapter for **OpenAudio S1 Mini** (by Fish Audio), a multilingual TTS model distilled from the larger S1 model. It uses a **Dual-AR Transformer** backbone (Qwen-based) and supports rich prompt engineering for emotion control.

## 🧠 Model Overview

| Feature | Details |
| :--- | :--- |
| **Architecture** | Dual-AR Transformer (Text-to-Semantic) + VQ-GAN (Semantic-to-Waveform). |
| **Model Size** | **~500M Parameters**. |
| **Languages** | **13+** (English, Chinese, Japanese, German, etc.). |
| **Voice Cloning** | **Yes**. Requires ~10-30s reference audio + **reference text**. |
| **Emotion Control** | **Text Markers**. Supports tags like `(laughing)`, `(sad)`, `(shouting)`. |
| **Codec** | Custom DAC/VQ-GAN. |

-----

## ⚙️ Installation

```bash
cd adapters/openaudio_s1mini
uv sync
````

*Installs `fish-speech` dependencies. Requires `hydra` for config management.*

-----

## 💻 Usage: Local Python

S1 Mini requires the transcript of the reference audio for optimal cloning quality. It also supports "Prompt Engineering" in the input text to trigger emotions.

### Basic Example

```python
from tts_adapter_openaudio_s1mini.adapter import OpenAudioS1MiniAdapter

# 1. Initialize
# Config paths are relative to the project structure
tts = OpenAudioS1MiniAdapter(
    llama_checkpoint_dir="checkpoints/openaudio-s1-mini",
    codec_checkpoint_path="checkpoints/openaudio-s1-mini/codec.pth",
    device="cuda"
)
tts.load_model()

# 2. Clone Voice (Audio + Text)
tts.clone_voice(
    ref_audio="data/ref/basic_ref_en.wav",
    ref_text="Some call me nature, others call me mother nature."
)

# 3. Synthesize with Emotion Markers
text = "(shouting) Hey you! (whispering) Come over here."
audio = tts.synthesize(text, temperature=0.7)

# 4. Long Text (Iterative)
# Use chunk_length > 0 to generate long content iteratively
audio_long = tts.synthesize(
    "Long text goes here...", 
    chunk_length=100
)
```

### Synthesis Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `temperature` | `0.8` | Controls randomness. Lower is more stable. |
| `top_p` | `0.8` | Nucleus sampling probability. |
| `repetition_penalty` | `1.1` | Prevents the model from getting stuck in loops. |
| `chunk_length` | `0` | If `>0`, enables iterative generation for long texts (chunk size in tokens). |
| `max_new_tokens` | `0` | `0` allows the model to determine length automatically. |

-----

## 🌐 Usage: API

**POST** `http://localhost:7000/v1/tts`

```json
{
  "adapter": "openaudios1mini",
  "init": { "device": "cuda" },
  "load_model": {},
  "clone_voice": {
    "ref_audio": { "name": "ref.wav", "b64": "..." },
    "ref_text": "Transcript of reference audio."
  },
  "synthesize": {
    "text": "(laughing) This is so cool!",
    "kwargs": { "temperature": 0.7 }
  }
}
```