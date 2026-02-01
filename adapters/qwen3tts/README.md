# Qwen3-TTS Adapter (TTS Playground)

This adapter integrates **Qwen3-TTS** into the TTS Playground “one-adapter-per-venv” architecture.

Qwen3-TTS is an instruction-driven, multilingual TTS family with **three main usage modes**:
1) **Voice Clone** (Base / VC): clone a voice from a short reference clip (with an optional transcript).
2) **Voice Design**: synthesize a brand-new voice from a natural-language description (“persona / timbre / style”).
3) **Custom Voice**: pick from a set of **preset premium voices** and steer with an instruction prompt.

Upstream resources:
- GitHub: QwenLM/Qwen3-TTS
- HF model family: Qwen/Qwen3-TTS-* (Base, VoiceDesign, CustomVoice)

---

## Adapter ID (Playground)

This README assumes your registry entry is:

- **adapter id**: `qwen3tts`
- **adapter module**: `tts_adapter_qwen3tts.adapter`
- **class**: `Qwen3TTSAdapter`

Add to `src/tts_playground/runtime/runner_manager.py`:

```py
ADAPTER_REGISTRY["qwen3tts"] = ("qwen3tts", "tts_adapter_qwen3tts.adapter", "Qwen3TTSAdapter")
```

---

## Install (adapter venv)

From repo root:

```bash
cd adapters/qwen3tts
uv sync
```

This creates `adapters/qwen3tts/.venv` and installs:
- `qwen-tts` (PyPI) – Qwen3-TTS runtime
- `torch`, `torchaudio`
- `numpy`, `soundfile`
- `tts-core`

> Note: upstream recommends Python 3.12 for the cleanest dependency experience, but the adapter can work on other versions depending on your Torch build.

---

## Model selection

Qwen3-TTS ships multiple checkpoints. Common HF repo ids:

- Base (Voice Clone): `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- VoiceDesign: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- CustomVoice: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`

There are also smaller 0.6B variants in the same naming scheme.

In the Playground adapter, you typically choose via `init.model_id`.

Example (API client / Docker):
```py
base_init = {
  "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "device": "cuda:0",
  "dtype": "bfloat16",
  "attn_implementation": "flash_attention_2",
}
```

---

## Key concepts (the “three modes”)

### 1) Voice Clone (Base)

You provide:
- `ref_audio` (a short clip, ideally clean, ~3–10 seconds)
- optionally `ref_text` (the transcript of that clip)

Two common strategies:
- **ICL (best)**: use both `ref_audio` + `ref_text`
- **x-vector only (no transcript)**: set `x_vector_only_mode=True` (lower fidelity / less style transfer, but no transcript required)

The adapter is designed to cache a “voice clone prompt” so you can:
- call `clone_voice(...)` once
- then synthesize multiple times without recomputing prompt features

### 2) Voice Design

You provide:
- `instruct`: a natural-language “persona” description (gender/age/tone/accent/energy/etc.)
- `text` + `language`

This produces speech in a newly designed voice. You can also “design then clone”:
1) generate a short “reference clip” with VoiceDesign
2) feed that clip into Voice Clone (Base) to reuse the designed voice across many lines

### 3) Custom Voice

You provide:
- `speaker`: one of the preset voice names (e.g. `Ryan`, `Vivian`, `Sohee`, …)
- optional `instruct` to steer style/pace/emotion
- `text` + `language`

This mode is great when you want consistent, high quality output without reference audio.

---

## Parameters you will tweak most

These are typically passed inside `synthesize.kwargs`:

### Universal
- `language`: `"Auto"` or `"English"`, `"Chinese"`, `"French"`, etc.
- `instruct`: steering text (“warm, calm, slower pace”, etc.)
- `temperature`, `top_p`, `top_k`, `do_sample`, `num_beams`, `repetition_penalty`, `max_new_tokens`

### Voice Clone only
- `x_vector_only_mode` (when building the prompt)
- `ref_text` (if using ICL)
- `voice_clone_prompt` (advanced: reuse a prebuilt prompt; the adapter usually caches this internally)

### Custom Voice only
- `speaker`: preset voice id/name

---

## Local usage (no API)

Run the adapter-local example scripts:

```bash
cd adapters/qwen3tts
uv run python examples/run_local.py
```

These scripts save outputs under `data/local_examples/qwen3tts/` (repo root).

---

## API usage (Docker stack)

1) Start the stack:

```bash
docker compose up --build
```

2) Run the client demo:

```bash
python examples/api_demo_qwen3tts.py
```

Outputs will be written under:
`data/api_examples/qwen3tts/`

---

## Troubleshooting

- **CUDA OOM**: try `dtype="float16"` or reduce `max_new_tokens`. Also try fewer concurrent tasks (`CELERY_CONCURRENCY=1`).
- **Slow first run**: HF downloads models the first time; keep your HF cache mounted (`HF_CACHE_HOST_DIR`).
- **Voice clone sounds off**: for best results, provide accurate `ref_text` matching the reference audio; otherwise use `x_vector_only_mode=True`.
- **FlashAttention errors**: set `attn_implementation="sdpa"` (or omit it) if FlashAttention isn’t available.
