# VibeVoice Adapter (tts-adapter-vibevoice)

Adapter for the **community VibeVoice** models (e.g. `vibevoice/VibeVoice-1.5B`, `vibevoice/VibeVoice-7B`).  
VibeVoice supports **multi-speaker scripts** and **voice conditioning (“prefill”)** from reference WAVs.

This adapter works in two ways:
- **Local Python** (import and run `VibeVoiceAdapter` directly)
- **TTS Playground API** (adapter id: `vibevoicetts`)

---

## Requirements

- **Python**: 3.12+
- **CUDA GPU recommended** (7B is heavy). CPU / Apple Silicon (MPS) may work but will be slow.
- **Hugging Face cache**: models are downloaded automatically on first run.

---

## Install (adapter venv)

```bash
cd adapters/vibevoice
uv sync
```

> If you run the Docker worker, the adapter runner will automatically create this venv the first time you call the adapter.

---

## Local usage

A minimal example (single speaker):

```python
from tts_adapter_vibevoice import VibeVoiceAdapter

tts = VibeVoiceAdapter(
    model_id="vibevoice/VibeVoice-7B",   # or "vibevoice/VibeVoice-1.5B"
    device=None,                         # auto: cuda > mps > cpu
    torch_dtype=None,                    # auto: bf16 on CUDA if supported, else fp16; fp32 on CPU/MPS
    attn_implementation=None,            # auto: flash_attention_2 on CUDA (SDPA fallback)
    cfg_scale=1.3,
    ddpm_steps=10,
    is_prefill=True,
    generation_config={"do_sample": False},
)

tts.load_model()

# Optional: voice conditioning ("prefill")
tts.clone_voice(ref_audio="data/ref/basic_ref_en.wav")

wav_bytes = tts.synthesize("Hello! This is VibeVoice via TTS Playground.")
open("out.wav", "wb").write(wav_bytes)
```

You can also run the provided script:

```bash
python adapters/vibevoice/examples/run_local.py
```

---

## Multi-speaker scripts

Use `Speaker N:` labels, for example:

```text
Speaker 1: Hello!
Speaker 2: Hi there.
Speaker 1: Nice to meet you.
```

### Providing voices

You can provide reference voices in two ways:

1) **Explicit map** (recommended)

```python
tts.clone_voice(
    speaker_voices={
        "1": "data/ref/basic_ref_en.wav",
        "2": "data/ref/fr/Ellie_Bishop_fr.wav",
    }
)
wav = tts.synthesize(script)
```

2) **Ordered list**

```python
tts.clone_voice(
    voice_samples=[
        "data/ref/basic_ref_en.wav",      # Speaker 1
        "data/ref/fr/Ellie_Bishop_fr.wav" # Speaker 2
    ]
)
wav = tts.synthesize(script)
```

### Speaker id normalization

To match VibeVoice’s upstream prompting logic, the adapter **normalizes speaker ids** internally:
- Speakers are remapped to `Speaker 1..N` in order of **first appearance**.
- Any `speaker_voices={...}` keys are remapped the same way.

This makes scripts like `Speaker 2` then `Speaker 1` (or non-contiguous ids) behave consistently.

---

## Key parameters

### Adapter init parameters (`VibeVoiceAdapter(...)`)

- `model_id`: HF repo id or a local directory
- `device`: `"cuda" | "mps" | "cpu" | None`
- `torch_dtype`: `"bfloat16" | "float16" | "float32" | None`
- `attn_implementation`: `"flash_attention_2" | "sdpa" | None`
- `cfg_scale`: classifier-free guidance scale (typical: ~1.0–2.0)
- `ddpm_steps`: diffusion refinement steps (quality vs speed)
- `is_prefill`: default behavior for `synthesize()`
- `generation_config`: Hugging Face generation kwargs (e.g. `do_sample`, `temperature`, `top_p`)
- `lora_checkpoint`: optional LoRA assets directory (if you fine-tuned VibeVoice)

### Per-call overrides (`synthesize(text, **kwargs)`)

- `cfg_scale`, `ddpm_steps`
- `is_prefill` (when `False`, the adapter **does not send any voice samples** to the processor)
- `generation_config`
- `seed`
- `speaker_voices` / `voice_samples` (per-call override)

---

## TTS Playground API usage

Adapter id (payload.adapter): **`vibevoicetts`**

Example request:

```json
{
  "adapter": "vibevoicetts",
  "init": { "model_id": "vibevoice/VibeVoice-7B", "device": "cuda" },
  "load_model": {},
  "clone_voice": {
    "ref_audio": { "name": "ref.wav", "b64": "..." }
  },
  "synthesize": {
    "text": "Speaker 1: Hello!",
    "kwargs": { "cfg_scale": 1.3, "generation_config": { "do_sample": false } }
  }
}
```

Multi-speaker clone via API (note: each value is a `{name,b64}` blob):

```json
{
  "clone_voice": {
    "speaker_voices": {
      "1": { "name": "en.wav", "b64": "..." },
      "2": { "name": "fr.wav", "b64": "..." }
    }
  }
}
```

---

## Troubleshooting

- **FlashAttention2 errors**: set `attn_implementation="sdpa"` or install FlashAttention2.
- **bf16 not supported**: the adapter auto-falls back to fp16 on CUDA.
- **VRAM**: try the **1.5B** model first.
