# TTS Playground

A lightweight **orchestrator + API** that lets you run multiple TTS models behind a **single HTTP endpoint**.
Each model lives in its own **adapter** folder (its own `uv` project + virtualenv). The worker spawns the right adapter on demand.

> Sample page: https://Nicolas-Prevot.github.io/TTS_playground/tts_samples.html

---

## What’s in this repo

- **API** (FastAPI): accepts a synthesis request and returns a Celery task id.
- **Worker** (Celery): runs the task, starts the requested adapter runner, and returns a WAV (base64).
- **Adapters** (`./adapters/<name>`): one `uv` project per model (`pyproject.toml`, `uv.lock`, `src/tts_adapter_*/adapter.py`).
- **Checkpoints** (`./checkpoints`): model weights/configs expected by some adapters (mounted into the worker container).
- **Configs** (`./configs`): YAML configs used by FishSpeech/OpenAudio-style adapters.

---

## Supported adapters (current registry)

The server exposes the following adapter ids (these are the values to send in `payload.adapter`):

- `chatterbox`
- `dia2`
- `f5tts`
- `fishspeech15`
- `higgsaudio`
- `indextts2`
- `kokoro`
- `kyutai`
- `openaudios1mini`
- `vibevoicetts`

Adapter implementations live in `./adapters/`.

---

## Quickstart (Docker)

### 1) Prereqs

- Docker + Docker Compose
- (Optional) NVIDIA GPU + NVIDIA Container Toolkit if you want CUDA acceleration.

### 2) Create your `.env`

Copy an example (recommended) and edit paths:

- `HF_CACHE_HOST_DIR` must be a **host** folder (where HuggingFace cache is stored).
- Windows users: prefer forward slashes, e.g. `C:/Users/you/.cache/huggingface`.

### 3) Start the stack

```bash
docker compose up -d --build
```

API will be available at:

- `POST http://localhost:7000/v1/tts`
- `GET  http://localhost:7000/v1/tasks/<task_id>`

### 4) Minimal API test (curl)

```bash
curl -X POST "http://localhost:7000/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{"adapter":"kokoro","init":{},"load_model":{},"clone_voice":{"voice":"af_bella","lang_code":"a"},"synthesize":{"text":"Hello from TTS Playground","kwargs":{}}}'
```

Then poll:

```bash
curl http://localhost:7000/v1/tasks/<task_id>
```

If `state == "SUCCESS"`, the response contains `wav_b64` and `sr`.

---

## Running the provided Python examples

Examples are under `./examples/` and use `TTSClient` (HTTP client).

1) Start Docker stack (previous section)
2) In another terminal (host), run:

```bash
python examples/api_demo_kokoro.py
```

Each example writes WAVs under `./data/api_examples/<adapter>/`.

---

## Adapter setup notes (important)

Some upstream repos include code that you may **not want to vendor** in this repository.
For those adapters, you must fetch/copy a few folders yourself.

### HiggsAudio: `audio_processing` folder is REQUIRED

The HiggsAudio adapter expects:

```
adapters/higgsaudio/src/tts_adapter_higgsaudio/audio_processing/
```

If you don’t commit it, the worker will refuse to start the adapter and will tell you which path is missing.

**How to populate it**

Option A (simple clone + copy):

```bash
git clone --depth 1 https://github.com/boson-ai/higgs-audio.git /tmp/higgs-audio
# Copy the folder from the upstream repo into this adapter:
cp -r /tmp/higgs-audio/boson_multimodal/audio_processing   adapters/higgsaudio/src/tts_adapter_higgsaudio/audio_processing
```

Option B (sparse checkout only that folder):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/boson-ai/higgs-audio.git /tmp/higgs-audio
cd /tmp/higgs-audio
git sparse-checkout set boson_multimodal/audio_processing
cp -r boson_multimodal/audio_processing   <PATH_TO_THIS_REPO>/adapters/higgsaudio/src/tts_adapter_higgsaudio/audio_processing
```

> Keep the LICENSE files from that folder (it contains third‑party code and attribution).

### IndexTTS2: checkpoints + any extra assets

IndexTTS2 expects its checkpoints under:

```
checkpoints/indextts2/
```

If you keep checkpoints out of git, download them separately (adapter README may have exact links),
then mount `./checkpoints` into the worker (already done in `docker-compose.yml`).

---

## Configuration knobs

All configuration is environment-driven (Docker `.env` is recommended):

- `CELERY_CONCURRENCY`: how many tasks the worker runs in parallel (default `1` to avoid VRAM pressure).
- `RUNNER_REQUEST_TIMEOUT`: max seconds to wait for a single adapter run.
- `IDLE_SECS` + `EXIT_ON_IDLE`: adapter runner exits after inactivity to free RAM/VRAM.
- `HF_CACHE_HOST_DIR`: host path for HuggingFace cache shared between API/worker.

---

## Development notes

### Adding a new adapter

1) Create `./adapters/<new_adapter>/` as its own `uv` project with `pyproject.toml` and `src/tts_adapter_<new_adapter>/adapter.py`.
2) Add the adapter to `src/tts_playground/runtime/runner_manager.py` in `ADAPTER_REGISTRY`.
3) If the adapter needs “non-vendored” folders, add them to `REQUIRED_PATHS` so the worker fails fast with a helpful message.

---

## Known TODOs

- Automatically download checkpoints / artifacts
- Improve sample page + add more comparisons
