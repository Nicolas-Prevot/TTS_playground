# IndexTTS2 Adapter

This directory contains the adapter for **IndexTTS2**, a robust, high-fidelity text-to-speech system. It uses a GPT-style autoregressive model to predict audio tokens (semantic/acoustic) and utilizes a **BigVGAN** vocoder for high-quality waveform generation. It is particularly notable for its advanced emotion control capabilities.

## 🧠 Model Overview

| Feature | Details |
| :--- | :--- |
| **Architecture** | Autoregressive Transformer + Conformer Reference Encoder + BigVGAN Vocoder. |
| **Model Size** | **~300M Parameters**. |
| **Languages** | **English & Chinese**. |
| **Voice Cloning** | **Yes**. Requires ~5–10s of reference audio. |
| **Emotion Control** | **Advanced**. Supports 3 modes: Reference Audio, Emotion Vector, and Natural Language Description. |
| **Sample Rate** | 22,050 Hz. |

-----

## ⚙️ Installation

Navigate to the adapter directory and sync dependencies using `uv`.

```bash
cd adapters/indextts2
uv sync
````

*Note: This adapter requires the `indextts` package, which is installed from the official repository via the `pyproject.toml`.*

-----

## 💻 Usage: Local Python

IndexTTS2 offers the most granular control over emotion among the adapters in this playground.

### Basic Example

```python
from tts_adapter_indextts2.adapter import IndexTTS2Adapter

# 1. Initialize
tts = IndexTTS2Adapter(
    model_dir="checkpoints/indextts2",
    cfg_path="checkpoints/indextts2/config.yaml",
    device="cuda"
)
tts.load_model()

# 2. Clone Voice (Sets the Speaker Timbre)
tts.clone_voice("data/ref/basic_ref_en.wav")

# 3. Synthesize (Neutral/Default Speaker Emotion)
audio = tts.synthesize("Hello, this is a standard generation.")

# 4. Advanced Emotion Controls

# Method A: Emotion from Audio (Style Transfer)
# Transfer the emotion of 'angry.wav' to the speaker's voice
audio_angry = tts.synthesize(
    "I am very upset right now!",
    emo_audio_prompt="data/ref/emo_hate.wav",
    emo_alpha=0.8 # Strength of transfer
)

# Method B: Emotion Vector (8-dimensional)
# [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
# Example: Mix of Happy (0.6) and Calm (0.6)
audio_vec = tts.synthesize(
    "I am feeling quite relaxed and happy.",
    emo_vector=[0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6],
    emo_alpha=1.0,
    use_random=False
)

# Method C: Text Description (QwenEmotion)
audio_desc = tts.synthesize(
    "It was a dark and stormy night...",
    use_emo_text=True,
    emo_text="Scared, whispering, slightly trembling",
    emo_alpha=0.9
)
```

### Synthesis Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `temperature` | `0.8` | Sampling temperature. |
| `top_p` | `0.9` | Nucleus sampling probability. |
| `interval_silence` | `200` | Silence (in ms) inserted between segmented sentences. |
| `emo_alpha` | `1.0` | Strength of the applied emotion (vector/reference/text). |
| `use_random` | `False` | If True, adds random noise to the emotion embedding for variation. |

-----

## 🌐 Usage: API

**POST** `http://localhost:7000/v1/tts`

```json
{
  "adapter": "indextts2",
  "init": { 
      "model_dir": "/workspace/checkpoints/indextts2",
      "cfg_path": "/workspace/checkpoints/indextts2/config.yaml" 
  },
  "load_model": {},
  "clone_voice": {
    "ref_audio": { "name": "ref.wav", "b64": "..." }
  },
  "synthesize": {
    "text": "I am happy!",
    "kwargs": {
      "emo_vector": [1.0, 0, 0, 0, 0, 0, 0, 0],
      "use_random": false
    }
  }
}
```