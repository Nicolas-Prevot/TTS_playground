# Dia2 Adapter

Adapter for **Dia2-2B** (Nari Labs), a dialogue-focused, low-latency TTS system designed for **turn-taking conversations**.

Dia2 expects **dialogue scripts** with speaker tags like:

```text
[S1] Hello!
[S2] Hi — great to meet you.
```

It also supports **prefix audio prompts** per speaker to steer timbre/style (Dia2 uses Whisper to transcribe these prompts, so conditioning adds latency).

---

## Installation (local clone required)

This adapter installs the `dia2` Python package from **local source** (editable mode) via `external/dia2`.

### 1) Clone upstream Dia2 into `external/dia2`

From the **root** of `TTS_playground`:

```bash
mkdir -p external
git clone --depth 1 https://github.com/nari-labs/dia2.git external/dia2
```

Alternative (recommended if you want it tracked but not vendored): use a git submodule:

```bash
git submodule add https://github.com/nari-labs/dia2.git external/dia2
git submodule update --init --recursive
```

### 2) Create the adapter environment

```bash
cd adapters/dia2
uv sync
```

This creates `adapters/dia2/.venv` and installs:
- this adapter (`tts-adapter-dia2`)
- `dia2` from `../../external/dia2` (editable)
- runtime deps (`torch`, `torchaudio`, `soundfile`, …)

> **Docker note**: `docker-compose.yml` mounts `./external` to `/workspace/external` in the worker container.  
> The adapter’s `uv` config uses `../../external/dia2`, so the clone must exist at `./external/dia2` on the host.

---

## Run the local example

From `adapters/dia2`:

```bash
uv run python examples/run_local.py
```

Outputs are written to:

```text
data/local_examples/dia2/
```

---

## Use in Python

```python
from tts_adapter_dia2.adapter import Dia2Adapter

tts = Dia2Adapter(
    repo_id="nari-labs/Dia2-2B",
    device=None,          # auto-select cuda if available
    dtype="bfloat16",     # auto-fallback to float32 on CPU
    cfg_scale=6.0,        # matches upstream CLI quickstart
    audio_temperature=0.8,
    audio_top_k=50,
    use_cuda_graph=True,  # only applied on CUDA
)
tts.load_model()

# Optional: prefix audio for conditioning (style steering)
tts.clone_voice(
    prefix_speaker_1="data/ref/basic_ref_en.wav",
    include_prefix=False,   # do NOT prepend reference audio to output
)

script = (
    "[S1] Hello, this is Dia2 running inside TTS Playground.\n"
    "[S2] Nice! We can generate natural dialogue directly from a script."
)

wav_bytes = tts.synthesize(script)
open("output_dia2.wav", "wb").write(wav_bytes)
```

---

## Important parameters

### Adapter init (`Dia2Adapter(...)`)

| Parameter | Default | Notes |
|---|---:|---|
| `repo_id` | `nari-labs/Dia2-2B` | HF weights repo |
| `device` | `None` | auto-select `cuda` if available |
| `dtype` | `bfloat16` | falls back to `float32` on CPU |
| `cfg_scale` | `6.0` | classifier-free guidance strength |
| `audio_temperature` | `0.8` | audio sampling temperature |
| `audio_top_k` | `50` | audio sampling top-k |
| `use_cuda_graph` | `True` | only used when `device=="cuda"` |

### Synthesis (`synthesize(...)`)

| Parameter | Default | Notes |
|---|---:|---|
| `temperature` | init default | overrides `audio_temperature` |
| `top_k` | init default | overrides `audio_top_k` |
| `cfg_scale` | init default | overrides init cfg |
| `prefix_speaker_1/2` | cached | pass `None` to disable cached prefix for one call |
| `include_prefix` | cached | if `True`, output starts with the prefix audio |
| `include_prefix_audio` | alias | accepted for backwards compatibility |

> Upstream keyword is `include_prefix`. Some older builds used `include_prefix_audio`.  
> This adapter accepts both; `include_prefix` wins if both are provided.

---

## API usage (via Docker stack)

1) Start the stack:

```bash
docker compose up --build
```

2) Run the provided client example:

```bash
python examples/api_demo_dia2.py
```

Outputs are written to:

```text
data/api_examples/dia2/
```

---

## Troubleshooting

- **Worker error: missing external paths**
  - The worker validates that `external/dia2/pyproject.toml` exists before starting the Dia2 runner.
  - Fix by cloning Dia2 into `./external/dia2` (see Installation section).

- **CPU + bfloat16**
  - If you run on CPU, the adapter automatically falls back from `bfloat16` to `float32`.

- **Prefix prompts are slow**
  - Dia2 uses Whisper to transcribe prefix audio, so conditioning adds noticeable latency.

---

## Credits & License

- Upstream: Nari Labs Dia2 — https://github.com/nari-labs/dia2
- Weights: https://huggingface.co/nari-labs/Dia2-2B
- License: Apache-2.0 (see upstream repository)
