# Chatterbox TTS Adapter

Adapter for **Resemble AI — Chatterbox (English)** via the `chatterbox-tts` Python package.

This adapter integrates with the **TTS_playground** runtime (FastAPI + Celery + one isolated `uv` env per adapter).

---

## Why this adapter is implemented this way

When you call the Playground API with a reference audio file, the worker **stages that upload into a per-request temporary directory** and deletes it after the request finishes.

Upstream Chatterbox supports voice cloning in two ways:

1) `model.generate(text, audio_prompt_path=...)` (it calls `prepare_conditionals()` internally), or  
2) call `model.prepare_conditionals(audio_prompt_path)` once, then call `model.generate(text)` without passing a path.

This adapter uses option (2) so it can **cache voice conditionals in memory** and keep synthesizing even after the original temp file is gone.

(Internally, Chatterbox stores conditionals in `model.conds`, and `generate()` reuses them when `audio_prompt_path` is not provided.)

---

## Requirements

- Python **3.12** (this adapter pins `requires-python = >=3.12,<3.13`)
- `uv` (recommended) or any PEP-517 compatible installer
- For GPU: a CUDA-enabled PyTorch build + NVIDIA drivers/toolkit (or use `device="cpu"`)

> Chatterbox-tts itself supports Python 3.10+ and downloads weights from Hugging Face on first use.

---

## Install (standalone adapter environment)

From the repo root:

```bash
cd adapters/chatterbox
uv sync
```

Run the local example:

```bash
python examples/run_local.py
```

The example expects a reference WAV at:

```text
data/ref/basic_ref_en.wav
```

and writes outputs to:

```text
data/local_examples/chatterbox/
```

---

## Using the adapter directly (Python)

```python
from tts_adapter_chatterbox.adapter import ChatterboxTTSAdapter

tts = ChatterboxTTSAdapter(device="cuda")  # "cuda", "mps", or "cpu"
tts.load_model()

# Optional: voice cloning. If skipped, Chatterbox uses its built-in default voice
# (the pretrained bundle includes a `conds.pt`).
tts.clone_voice("data/ref/basic_ref_en.wav")

audio_bytes = tts.synthesize(
    "I've been a silent spectator, watching empires rise and fall.",
    cfg_weight=0.5,
    exaggeration=0.6,
    temperature=0.8,
)

with open("output_chatterbox.wav", "wb") as f:
    f.write(audio_bytes)
```

---

## Using via the Playground API

### Key behavior: caching is per-runner-process

Voice conditionals are cached inside the **adapter runner process**.

They will be lost if:
- the runner exits due to the idle timeout (`IDLE_SECS` + `EXIT_ON_IDLE`), or
- the Playground switches to a different adapter (the Manager stops the previous runner to free RAM/VRAM).

So if you want a “sticky” voice, keep using the same adapter and increase `IDLE_SECS`.

### Python client example

```python
from tts_playground.client.tts_client import TTSClient

client = TTSClient("http://localhost:7000", timeout=300.0)

ref_blob = client.pack_file("data/ref/basic_ref_en.wav")

# 1) First request: clone voice (uploads ref audio once)
client.synth(
    adapter="chatterbox",
    init={"device": "cuda"},
    load_model={},
    clone_voice={"ref_audio": ref_blob},
    synthesize={"text": "Hello from Chatterbox.", "kwargs": {"exaggeration": 0.6}},
    wait=True,
    download=True,
    dest_path="out_01.wav",
)

# 2) Next request: reuse cached conditionals
# IMPORTANT: pass an empty object for clone_voice ({}). The API schema always includes it.
client.synth(
    adapter="chatterbox",
    init={"device": "cuda"},
    load_model={},
    clone_voice={},  # reuse the last cloned voice in this runner process
    synthesize={"text": "Second line, same voice.", "kwargs": {"exaggeration": 0.5}},
    wait=True,
    download=True,
    dest_path="out_02.wav",
)

client.close()
```

---

## Reference audio tips

- Chatterbox can clone a voice from **a few seconds** of reference audio.
- In practice, **5–20 seconds** tends to work well depending on noise/clarity.

Use a clean, single-speaker clip (minimal music/background).

---

## Parameters exposed by this adapter

These map to `ChatterboxTTS.generate(...)`:

| Parameter | Default | Meaning |
|---|---:|---|
| `exaggeration` | `0.5` | Emotion intensity (higher = more expressive; can also speed up delivery). |
| `cfg_weight` | `0.5` | CFG strength. Lower values can help pacing for fast reference voices. |
| `temperature` | `0.8` | Sampling randomness. |
| `top_p` | `1.0` | Nucleus sampling cutoff. |
| `min_p` | `0.05` | Probability floor used by the sampler. |
| `repetition_penalty` | `1.2` | Helps reduce repetition (“stuttering”). |

---

## Output details

- Sample rate: **24 kHz** (Chatterbox `model.sr`).
- Audio is **watermarked** with Resemble AI’s PerTh watermarker (built into Chatterbox).

---

## Troubleshooting

### “Torch not compiled with CUDA enabled”
You installed a CPU-only PyTorch build. Either:
- reinstall torch/torchaudio with CUDA wheels that match your CUDA runtime, or
- set `device="cpu"`.

This adapter will automatically fall back to CPU if you request `"cuda"` but CUDA is not available.

### First run is slow
The first run downloads model weights into your Hugging Face cache (`HF_HOME` / `HF_HUB_CACHE`).

---

## Upstream links / attribution

- Upstream repo: https://github.com/resemble-ai/chatterbox
- PyPI: https://pypi.org/project/chatterbox-tts/
- Hugging Face model: https://huggingface.co/ResembleAI/chatterbox

License and weight terms come from upstream (MIT for the code; always verify before deployment).
