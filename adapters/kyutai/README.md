# Kyutai (Moshi) TTS Adapter

This directory contains the adapter for **Kyutai TTS** (specifically the `tts-1.6b-en_fr` model), part of the Moshi real-time audio project. It uses a streaming-native architecture with the **Mimi** neural audio codec.

## 🧠 Model Overview

| Feature | Details |
| :--- | :--- |
| **Architecture** | Transformer-based (Moshi) utilizing the Mimi audio codec. |
| **Model Size** | **1.6B Parameters**. |
| **Languages** | **English & French**. |
| **Voice Cloning** | **Restricted**. Can only "clone" specific voice IDs provided in the Moshi/Kyutai repositories (e.g., `vctk/p225`, `cml-tts/...`). Arbitrary zero-shot cloning is not supported in this version. |
| **Streaming** | Native support (though this adapter returns full files). |

-----

## ⚙️ Installation

```bash
cd adapters/kyutai
uv sync
````

*Installs `moshi` and `torch`.*

-----

## 💻 Usage: Local Python

This adapter requires you to pass a specific "Voice ID" or a path to a `.wav` file that exists within the model's known voice bank.

### Basic Example

```python
from tts_adapter_kyutai.adapter import KyutaiTTSAdapter

# 1. Initialize
tts = KyutaiTTSAdapter(
    hf_repo="kyutai/tts-1.6b-en_fr",
    device="cuda",
    temp=0.6
)
tts.load_model()

# 2. Select Voice
# Using a VCTK ID recognized by the model
tts.clone_voice("vctk/p225_023.wav")

# 3. Synthesize
audio = tts.synthesize("This is Kyutai generating speech via the Mimi codec.")
```

### Synthesis Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `temp` | `0.6` | Sampling temperature. Controls randomness/expressiveness. |
| `cfg_coef` | `1.0` | Classifier-Free Guidance coefficient. |
| `n_q` | `32` | Number of quantizers for the Mimi codec. |

-----

## 🌐 Usage: API

**POST** `http://localhost:7000/v1/tts`

```json
{
  "adapter": "kyutai",
  "init": { "device": "cuda" },
  "load_model": {},
  "clone_voice": {
    "voice_sample": "vctk/p225_023.wav"
  },
  "synthesize": {
    "text": "Bonjour, comment allez-vous?",
    "kwargs": {}
  }
}
```