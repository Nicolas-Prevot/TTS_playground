# Kokoro TTS Adapter

This directory contains the adapter for **Kokoro**, an ultra-lightweight (82M parameter) TTS model based on StyleTTS2 and iSTFTNet. It is designed for extreme speed and efficiency while maintaining high-quality prosody.

## 🧠 Model Overview

| Feature | Details |
| :--- | :--- |
| **Architecture** | StyleTTS2 backbone + iSTFTNet Vocoder. |
| **Model Size** | **82M Parameters**. Runs fast on CPU. |
| **Languages** | **Multilingual** (US/UK English, French, Japanese, Chinese, etc.). |
| **Voice Cloning** | **No**. Uses fixed, pre-trained voice embeddings (e.g., `af_heart`, `bm_george`). |
| **Audio Quality** | 24 kHz sample rate. |
| **Speed** | Very Fast. |

-----

## ⚙️ Installation

```bash
cd adapters/kokoro
uv sync
````

*Depends on `kokoro`, `misaki`, and `torch`.*

-----

## 💻 Usage: Local Python

Kokoro does not clone voices from audio files. Instead, `clone_voice` is used to select one of the built-in voice packs.

### Basic Example

```python
from tts_adapter_kokoro.adapter import KokoroTTSAdapter

# 1. Initialize
# lang_code: 'a' (US English), 'b' (UK English), 'f' (French), 'j' (Japanese), 'z' (Chinese)
tts = KokoroTTSAdapter(lang_code="a", voice="af_heart")
tts.load_model()

# 2. Select Voice
# "af_heart" = American Female Heart
# "bm_george" = British Male George
tts.clone_voice(voice="bm_george", lang_code="b") 

# 3. Synthesize
audio = tts.synthesize(
    "Empires rise and fall, but I remain.",
    speed=1.0 # Adjust speaking rate
)
```

### Synthesis Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `speed` | `1.0` | Speaking rate. Higher is faster. |
| `split_pattern` | `\n+` | Regex pattern used to split text into chunks for generation. |

### Available Voices (Partial List)

  * **English (US):** `af_heart`, `af_bella`, `af_nicole`, `am_michael`
  * **English (UK):** `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`
  * **French:** `ff_siwis`
  * **Japanese:** `jf_tebukuro`

-----

## 🌐 Usage: API

**POST** `http://localhost:7000/v1/tts`

```json
{
  "adapter": "kokoro",
  "init": { "lang_code": "a", "voice": "af_heart" },
  "load_model": {},
  "clone_voice": {
    "voice": "af_bella",
    "lang_code": "a"
  },
  "synthesize": {
    "text": "Hello world.",
    "kwargs": { "speed": 1.2 }
  }
}
```