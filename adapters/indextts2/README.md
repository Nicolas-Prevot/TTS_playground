# IndexTTS2 Adapter (TTS Playground)

This adapter integrates **IndexTTS2** (IndexTeam) into the TTS Playground “one-adapter-per-venv” architecture.

- Upstream repo: **IndexTeam / index-tts** (IndexTTS-2)
- Key features: **zero-shot voice cloning**, and **emotion control** via:
  1) emotion reference audio (`emo_audio_prompt`)
  2) 8‑D emotion vector (`emo_vector`)
  3) natural-language emotion (`use_emo_text` + `emo_text`)

> Note: when using `emo_vector` or `use_emo_text`, upstream recommends setting `emo_audio_prompt=None`.
> This adapter enforces that automatically.

---

## Requirements

- Python: **≥ 3.10**
- GPU is strongly recommended for speed (CPU works but is slow).
- Enough disk space for model checkpoints + HuggingFace cache (several GB).

---

## Install (adapter venv)

From repo root:

```bash
cd adapters/indextts2
uv sync
```

This creates `adapters/indextts2/.venv` and installs:
- `indextts` from the upstream repo (configured in this adapter’s `pyproject.toml`)
- Torch + Torchaudio (this adapter is configured to use the PyTorch CUDA 12.8 index)

---

## Download checkpoints

This repo expects IndexTTS2 checkpoints under:

```
<repo_root>/checkpoints/indextts2/
```

### Recommended: HuggingFace CLI download

```bash
# install hf CLI (one-time, with uv)
uv tool install "huggingface-hub[cli,hf_xet]"

# from repo root:
hf download IndexTeam/IndexTTS-2 --local-dir checkpoints/indextts2
```

After download, you should have (names may evolve, but these are commonly present):

- `config.yaml`
- `gpt.pth`
- `s2mel.pth`
- `bpe.model`
- `wav2vec2bert_stats.pt`
- `feat1.pt`, `feat2.pt`
- `qwen0.6bemo4-merge/…`
- (optional but often included) `pinyin.vocab` (needed for pinyin-related features)

### About extra auto-downloads

IndexTTS2 may also pull auxiliary weights at first run.
In general, keep your HuggingFace cache (`HF_HOME` / `HF_HUB_CACHE`) writable and persistent.

#### ⚠️ HuggingFace cache path (upstream forces a local cache)

The upstream **`indextts`** package sets the environment variable:

- `HF_HUB_CACHE=./checkpoints/hf_cache`

**at import time**.

This means **HuggingFace Hub downloads will *not* use your usual global HF cache** (e.g. `C:/Users/<you>/.cache/huggingface`)
even if you have that configured elsewhere.

What you’ll observe:

- If you run from `adapters/indextts2/`, it will create:
  - `adapters/indextts2/checkpoints/hf_cache/…`
- If you run from the repo root, it will create:
  - `checkpoints/hf_cache/…`

This behavior comes from upstream IndexTTS2, not from this adapter. It can lead to duplicate downloads and extra disk usage.

---

## Local usage (no API)

Run the adapter-local example (recommended):

```bash
cd adapters/indextts2
uv run python examples/run_local.py
```

By default it looks for:
- Speaker reference: `data/ref/basic_ref_en.wav`
- Optional emotion reference: `data/ref/emo_hate.wav`
- Checkpoints: `checkpoints/indextts2/`

You can override paths via CLI flags (see the script `--help`).

---

## API usage (Docker stack)

1) Start the stack from repo root:

```bash
docker compose up --build
```

2) Run the provided client example:

```bash
python examples/api_demo_indextts2.py
```

### Notes on API payloads

- `clone_voice.ref_audio` is a file “blob” (`{"name": "...", "b64": "..."}`) in the client examples.
- `synthesize.kwargs.emo_audio_prompt` can also be a blob; the worker stages blobs to temp files before calling the adapter.

---

## Key parameters (what you’ll actually tweak)

### Voice cloning
- `clone_voice(ref_audio=...)` sets the speaker reference (timbre). Use ~5–10s clean speech if possible.

### Emotion controls
- `emo_audio_prompt`: path/blob to *emotion* reference audio
- `emo_alpha` (0..1): how strongly emotion is applied
- `emo_vector`: 8‑D vector in this order:
  `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`
- `use_emo_text`: enable natural language emotion
- `emo_text`: description (if omitted while `use_emo_text=True`, the adapter uses the synthesis `text`)
- `use_random`: adds randomness to emotion embedding (more variety, often less faithful cloning)

### Sampling / decoding
Common knobs:
- `temperature`, `top_p`, `top_k`, `num_beams`
- `repetition_penalty`, `max_mel_tokens`

### Segmentation (long text)
IndexTTS2 splits text into sentences internally. Two useful knobs:

- `max_text_tokens_per_sentence` (default 120)
- `interval_silence` (ms, default 200): silence inserted between stitched sentences

> Back-compat: this adapter also accepts `max_text_tokens_per_segment` as an alias.

---

## Troubleshooting

- **Missing checkpoints / config**: confirm `checkpoints/indextts2/config.yaml` exists.
- **Unexpected `checkpoints/hf_cache` folder appears**: see the **“HuggingFace cache path gotcha”** note above (upstream forces `HF_HUB_CACHE=./checkpoints/hf_cache`).
- **Warning about `max_mel_tokens`**: reduce `max_text_tokens_per_sentence` or increase `max_mel_tokens`.
- **CUDA OOM**: try fp16, reduce `max_mel_tokens`, shorten text, or run fewer concurrent tasks.
- **Text emotion seems ignored**: ensure `use_emo_text=True`. If also passing `emo_audio_prompt`, note that `emo_vector` / `use_emo_text` overrides `emo_audio_prompt` by design.
