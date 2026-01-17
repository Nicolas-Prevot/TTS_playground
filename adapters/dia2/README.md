# Dia2 Adapter

This directory contains the adapter for **Dia2 2B**, a streaming dialogue text-to-speech system by **Nari Labs**. Unlike traditional TTS, Dia2 is optimized for generating natural, turn-taking conversations. It can start generating speech as soon as the first words arrive and uses audio prompts to steer voice and style.

## 🧠 Model Overview

| Feature | Details |
| :--- | :--- |
| **Architecture** | Streaming dialogue transformer. Separates text & audio sampling configs. Uses **Mimi codec** + CUDA graphs for low latency. |
| **Model Size** | **Dia2-2B** (This adapter defaults to the 2B checkpoint). |
| **Languages** | **English**. Designed for \~2 minutes of coherent audio per generation. |
| **Voice Conditioning** | **Yes (Style Prompting)**. Use short prefix audio clips per speaker (e.g., `prefix_speaker_1`). |
| **Emotion Control** | **Implicit**. Emotion & prosody are derived from the script context and audio prefix. |
| **Streaming** | **Yes**. Designed to begin generating audio tokens before the full text script is processed. |
| **License** | **Apache-2.0** (Same as upstream). |

-----

## ⚙️ Installation

This adapter runs as an isolated environment within `TTS_playground`. However, due to packaging issues in the upstream repository, the installation process differs slightly from other adapters.

### ⚠️ The "External" Folder Setup

If you try to install `dia2` directly from PyPI or a standard wheel, you will encounter a `ModuleNotFoundError: No module named 'dia2.core'`. This is because the current upstream build configuration excludes required submodules.

To fix this, we utilize a local source installation via an **`external/`** directory.

### 1\. Clone Dia2 Locally

From the root of `TTS_playground`, create the `external` folder and clone the repository there:

```bash
# From the project root
mkdir -p external
git clone https://github.com/nari-labs/dia2.git external/dia2
```

### 2\. Configure `uv`

The adapter is already configured to look for this folder. The `pyproject.toml` in this directory contains:

```toml
[tool.uv.sources]
dia2 = { path = "../../external/dia2", editable = true }
```

This tells `uv` to install Dia2 in "editable mode" from your local clone, ensuring all sub-packages (`dia2.core`, `dia2.audio`, etc.) are correctly discovered.

### 3\. Sync Environment

Navigate to the adapter directory and install dependencies:

```bash
cd adapters/dia2
uv sync
```

*This creates the virtual environment and installs `torch`, `torchaudio`, and the local `dia2` reference.*

-----

## 💻 Usage: Local Python

You can use the adapter directly in Python scripts. Dia2 is unique because it expects **dialogue scripts** (e.g., `[S1]`, `[S2]`) rather than plain text.

### Basic Example

```python
from tts_adapter_dia2.adapter import Dia2Adapter

# 1. Initialize & Load
tts = Dia2Adapter(
    repo_id="nari-labs/Dia2-2B",  # HF repo ID
    device="cuda",                # "cpu" is supported but slower
    dtype="bfloat16",             # Recommended for CUDA
    cfg_scale=2.0,
    use_cuda_graph=True,
)
tts.load_model()

# 2. (Optional) Audio Prompting
# This serves as a style/voice prompt rather than strict cloning.
tts.clone_voice(
    prefix_speaker_1="data/ref/basic_ref_en.wav",
    include_prefix_audio=False,   # Use for conditioning only; do not include in output
)

# 3. Prepare Dialogue Script
script = (
    "[S1] Hello, this is Dia2 running inside the Playground.\n"
    "[S2] Nice! We can generate natural conversation directly from a script."
)

# 4. Synthesize
audio_bytes = tts.synthesize(script)

with open("output_dia2.wav", "wb") as f:
    f.write(audio_bytes)
```

### Synthesis Parameters

The `synthesize` method exposes high-level knobs that map to Dia2’s internal configuration:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `temperature` | `0.8` | Shortcut for `temp_audio`. Higher values = more variation/expressiveness; Lower = safer/monotone. |
| `top_k` | `50` | Shortcut for `topk_audio`. Controls the sampling pool size. |
| `cfg_scale` | `2.0` | **Classifier-Free Guidance.** Higher values force the model to adhere strictly to the text. |
| `prefix_speaker_X` | `None` | Override the voice prompt for specific speakers per-call. |
| `include_prefix` | `False` | If `True`, the output audio will start with the reference audio clip. |

-----

## 🌐 Usage: API (Docker Compose)

The `TTS_playground` orchestrator can serve Dia2 via HTTP.

### 1\. Start the Stack

Ensure you have performed the **Installation** steps above (cloning into `external/dia2`), then:

```bash
docker compose up -d
```

### 2\. Python Client (`TTSClient`)

```python
from tts_playground.client.tts_client import TTSClient

client = TTSClient("http://localhost:7000")
ref_blob = client.pack_file("data/ref/basic_ref_en.wav")

script = (
    "[S1] Hello via the Dia2 API.\n"
    "[S2] Great, the adapter wiring works!"
)

result = client.synth(
    adapter="dia2",
    init={
        "repo_id": "nari-labs/Dia2-2B",
        "device": "cuda",
        "dtype": "bfloat16",
        "use_cuda_graph": True
    },
    load_model={},
    clone_voice={
        "prefix_speaker_1": ref_blob,
        "include_prefix_audio": False,
    },
    synthesize={
        "text": script,
        "kwargs": {"temperature": 0.9}
    },
    download=True,
    dest_path="api_output_dia2.wav"
)
```

### 3\. Direct HTTP Request

**POST** `http://localhost:7000/v1/tts`

```json
{
  "adapter": "dia2",
  "init": {
    "repo_id": "nari-labs/Dia2-2B",
    "device": "cuda",
    "dtype": "bfloat16",
    "use_cuda_graph": true
  },
  "load_model": {},
  "clone_voice": {
    "prefix_speaker_1": {
      "name": "ref.wav",
      "b64": "<base64_encoded_wav_bytes>"
    },
    "include_prefix_audio": false
  },
  "synthesize": {
    "text": "[S1] This is a raw HTTP request test.\n[S2] Dia2 responds in dialogue form.",
    "kwargs": {
      "temperature": 0.9,
      "top_k": 80
    }
  }
}
```

-----

## 🎙️ Best Practices

1.  **Dialogue First:** Dia2 is built for conversation. Write scripts using `[S1]`, `[S2]` tags rather than treating it as a single-speaker narrator.
2.  **Conditioning vs. Cloning:** The `prefix_speaker` audio acts as a style conditioner. It steers the timbre and prosody, but it is not a "strict" clone like F5-TTS or XTTS. The output voice may vary slightly between runs.
3.  **Prompt Length:** Short prompts (3-10 seconds) generally work best.
4.  **Temperature:** If the model mumbles or speaks gibberish, try lowering the `temperature` (e.g., to 0.6 or 0.7). If it sounds too robotic, increase it (e.g., 0.9).
5.  **Duration:** The 2B model is optimized for generations up to \~2 minutes. For longer content, split your script into smaller chunks.

## 🔗 Credits & License

  * **Original Model & Code:** [Nari Labs – Dia2](https://github.com/nari-labs/dia2)
  * **Model Weights:** [Hugging Face – nari-labs/Dia2-2B](https://huggingface.co/nari-labs/Dia2-2B)
  * **License:** Apache-2.0