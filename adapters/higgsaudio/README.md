# HiggsAudio Adapter (BosonAI Higgs Audio v2)

This adapter integrates **BosonAI Higgs Audio v2** into the **TTS_playground** orchestrator.

It supports:
- **Text → speech** (single speaker)
- **Zero/one-shot voice cloning** using **reference audio + transcript**
- Optional **scene prompting** via the model’s special tags


## Requirements

- **Python**: 3.10+ (this adapter is its own `uv` project)
- **GPU strongly recommended** (CPU works but is extremely slow for this class of model)


## Important: you must provide `audio_processing/` (not tracked in git)

The adapter imports helper code that lives in the upstream Higgs Audio repository under:

```
boson_multimodal/audio_processing/
```

To keep this repo clean, you can **keep it out of git** and copy it locally when needed.

### Expected path in this repo

```
adapters/higgsaudio/src/tts_adapter_higgsaudio/audio_processing/
```

The worker will refuse to start `higgsaudio` if that folder is missing (fail-fast).


### Populate `audio_processing/` (clone + copy)

```bash
# From repo root
git clone --depth 1 https://github.com/boson-ai/higgs-audio.git /tmp/higgs-audio

mkdir -p adapters/higgsaudio/src/tts_adapter_higgsaudio
cp -R /tmp/higgs-audio/boson_multimodal/audio_processing   adapters/higgsaudio/src/tts_adapter_higgsaudio/audio_processing
```

This folder contains third-party code and attribution. Keep the `LICENSE` files that come with it.

### Keep it out of git

Add this to your repo’s `.gitignore`:

```
adapters/higgsaudio/src/tts_adapter_higgsaudio/audio_processing/
```


## Install (adapter environment)

```bash
cd adapters/higgsaudio
uv sync
```


## Run locally (no API)

```bash
cd adapters/higgsaudio
uv run python examples/run_local.py
```

Outputs a WAV under:

```
data/local_examples/higgsaudio/higgs_output.wav
```


## Run via the Playground API (Docker)

1) Start the stack from repo root:

```bash
docker compose up --build
```

2) Run the provided client example from the host:

```bash
uv run python examples/api_demo_higgsaudio.py
```

Outputs WAVs under:

```
data/api_examples/higgsaudio/
```


## Usage notes (Higgs-specific)

### Scene prompting

The official prompt format uses the model’s special scene tags:

- `<|scene_desc_start|> ... <|scene_desc_end|>`

This adapter accepts `scene_prompt` and injects it into the system prompt using those tags.

Good prompts look like:
- “Audio is recorded from a quiet room.”
- “A calm narrator in a studio, close mic, minimal reverb.”
- “A whisper in a library.”

### Voice cloning quality depends on the transcript

For voice cloning, provide:
- **5–10s** reference audio, clean & single-speaker
- **Exact transcript** of that reference audio (`ref_text`)

Even small transcript mismatches can degrade results.

### Long text

Enable chunking for long text:

```python
tts.synthesize(
  text,
  chunk_method="word",
  chunk_max_word_num=100,
  generation_chunk_buffer_size=2,
)
```
