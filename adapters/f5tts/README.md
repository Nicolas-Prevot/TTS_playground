# F5-TTS Adapter

This directory contains the adapter for **SWivid’s F5-TTS / E2-TTS**, a fast non-autoregressive text-to-speech system that supports strong **zero-shot voice cloning**.

It is designed to run inside **TTS_playground** as an isolated `uv` environment and exposes a consistent interface (`load_model()`, `clone_voice()`, `synthesize()`).

---

## Model Overview

| Feature | Details |
| :--- | :--- |
| **Architecture** | Diffusion Transformer (DiT) backbone trained via Flow Matching. |
| **Model Size** | ~335M parameters (Base). |
| **Languages** | English + Chinese (Mandarin) are supported by upstream checkpoints. |
| **Voice Cloning** | **Yes (Zero-shot)** — best with ~10–20s reference audio **and** its transcript. |
| **Emotion Control** | Implicit — style follows the reference clip and script. |
| **Vocoder** | Upstream supports **Vocos** and (for specific variants) **BigVGAN**. |
| **Inference Speed** | Fast (non-autoregressive). |

---

## Installation

From the adapter directory:

```bash
cd adapters/f5tts
uv sync
```

This installs `f5-tts`, `torch`, `torchaudio`, and the adapter dependencies in an isolated virtual environment.

---

## Local Usage

### Basic Example

```python
from tts_adapter_f5tts.adapter import F5TTSAdapter

# 1) Init + load
tts = F5TTSAdapter(
    model_name="F5TTS_v1_Base",
    vocoder_name="vocos",
    device="cuda",
)
tts.load_model()

# 2) Clone voice (audio + transcript strongly recommended)
tts.clone_voice(
    ref_audio="data/ref/basic_ref_en.wav",
    ref_text="Some call me nature, others call me mother nature."
)

# 3) Synthesize
audio_bytes = tts.synthesize(
    "F5-TTS uses flow matching to generate speech quickly.",
    nfe_step=32,
    speed=1.0,
)

with open("output_f5tts.wav", "wb") as f:
    f.write(audio_bytes)
```

### Reference Transcript Note (Important)

F5-TTS cloning quality depends heavily on `ref_text` being accurate.
If you **omit** `ref_text`, upstream preprocessing may attempt to run Whisper ASR to transcribe the reference audio — this is slower and may require additional heavy dependencies / model downloads. For best results and fastest runtime, always pass the transcript.

---

## Synthesis Parameters

The `synthesize()` method forwards common knobs into upstream inference:

| Parameter | Typical | Meaning |
| :--- | :---: | :--- |
| `speed` | `1.0` | Speaking rate (`>1.0` faster, `<1.0` slower). |
| `nfe_step` | `32` | Denoising steps (fewer = faster, more = higher fidelity). |
| `cfg_strength` | `2.0` | Classifier-free guidance strength (higher = stricter text adherence). |
| `sway_sampling_coef` | `-1.0` | Sampling trajectory tweak (small negative values often stabilize). |
| `cross_fade_duration` | `0.2` | Overlap (seconds) when stitching long text chunks. |
| `target_rms` | `0.1` | Loudness normalization target. |
| `fix_duration` | `-1` | Force duration (advanced; usually leave default). |

---

## Vocoder Notes

- `vocoder_name="vocos"` is the most commonly supported setting.
- `vocoder_name="bigvgan"` is supported **only for certain checkpoint variants** (for example `F5TTS_Base_bigvgan`). If you switch vocoders without using the matching model variant, inference may degrade or fail.

---

## API Usage (Docker Compose)

The TTS_playground orchestrator can serve this adapter via HTTP.

### Python Client (`TTSClient`)

```python
from tts_playground.client.tts_client import TTSClient

client = TTSClient("http://localhost:7000")
ref_blob = client.pack_file("data/ref/basic_ref_en.wav")

result = client.synth(
    adapter="f5tts",
    init={
        "model_name": "F5TTS_v1_Base",
        "vocoder_name": "vocos",
        "device": "cuda"
    },
    load_model={},
    clone_voice={
        "ref_audio": ref_blob,
        "ref_text": "Some call me nature, others call me mother nature."
    },
    synthesize={
        "text": "Hello via API.",
        "kwargs": {"nfe_step": 32, "speed": 1.0}
    },
    download=True,
    dest_path="api_output.wav"
)
```

## Credits & License

- **Original Model / Code**: SWivid/F5-TTS
- **License**: MIT (upstream)
