# OpenAudio S1 Mini Adapter (`openaudios1mini`)

Adapter for **OpenAudio S1 Mini** from Fish Audio (0.5B “S1-mini” variant). The upstream model uses:
- a **Dual-AR Transformer** (text → semantic codes)
- a **DAC-based codec** (semantic codes → waveform)

This adapter calls Fish-Speech’s Python APIs directly (no subprocess).

---

## Requirements

- Python >= 3.12
- A CUDA GPU is recommended.
- On older GPUs (or when bf16 isn’t supported), run with **fp16** (`half=True`).  

---

## 1) Install (adapter venv)

```bash
cd adapters/openaudio_s1mini
uv sync
```

This creates `adapters/openaudio_s1mini/.venv` and installs `fish-speech` + the adapter.

---

## 2) Download checkpoints (recommended: HuggingFace Hub)

Place the OpenAudio S1 mini weights under:

```
checkpoints/openaudio-s1-mini/
  ├─ model.pth
  ├─ codec.pth
  ├─ config.json
  ├─ tokenizer.tiktoken
  └─ ...
```

You can download them from HF:

```bash
# Requires huggingface_hub >= 0.20 (provides the `hf` CLI)
hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

> In Docker, the worker container expects checkpoints at `/workspace/checkpoints/...` (already mounted by `docker-compose.yml`).

---

## 3) Codec config (Hydra)

This adapter loads the codec via Hydra config:

```
configs/openaudio-s1-mini/modded_dac_vq.yaml
```

You normally **don’t** need to change this unless you update Fish-Speech internals.

In Docker, the path is `/workspace/configs/openaudio-s1-mini`.

---

## 4) Local usage (no API)

A runnable example is provided:

```bash
python adapters/openaudio_s1mini/examples/run_local.py
```

It will write outputs to:

```
data/local_examples/openaudio_s1mini/
```

---

## 5) Programmatic usage

```python
from tts_adapter_openaudio_s1mini import OpenAudioS1MiniAdapter

tts = OpenAudioS1MiniAdapter(
    llama_checkpoint_dir="checkpoints/openaudio-s1-mini",
    codec_checkpoint_path="checkpoints/openaudio-s1-mini/codec.pth",
    decoder_config_name="modded_dac_vq",
    config_root_path="configs/openaudio-s1-mini",
    device="cuda",
    half=True,   # fp16 (recommended on older GPUs)
)

tts.load_model()

# Best quality when you provide the transcript of the reference audio:
tts.clone_voice(
    ref_audio="data/ref/basic_ref_en.wav",
    ref_text="Some call me nature, others call me mother nature.",
)

wav_bytes = tts.synthesize(
    "(laughing) This is a test!",
    temperature=0.7,
    top_p=0.9,
)
```

### Notes

- `clone_voice()` caches *prompt tokens* from the reference audio (via the codec).
- Supplying `ref_text` (transcript) typically improves voice cloning quality.
- Long text: pass `chunk_length>0` to enable iterative prompting.

---

## 6) API usage (TTS Playground)

**POST** `http://localhost:7000/v1/tts`

```json
{
  "adapter": "openaudios1mini",
  "init": {
    "llama_checkpoint_dir": "/workspace/checkpoints/openaudio-s1-mini",
    "codec_checkpoint_path": "/workspace/checkpoints/openaudio-s1-mini/codec.pth",
    "decoder_config_name": "modded_dac_vq",
    "config_root_path": "/workspace/configs/openaudio-s1-mini",
    "device": "cuda",
    "half": true
  },
  "load_model": {},
  "clone_voice": {
    "ref_audio": { "name": "ref.wav", "b64": "..." },
    "ref_text": "Transcript of reference audio."
  },
  "synthesize": {
    "text": "(shouting) Hello there!",
    "kwargs": { "temperature": 0.7, "top_p": 0.9 }
  }
}
```

---

## Troubleshooting

- **No bf16 support / dtype errors**: set `half=True` (fp16).
- **Missing checkpoint/config**: verify your paths:
  - host: `checkpoints/openaudio-s1-mini`
  - docker: `/workspace/checkpoints/openaudio-s1-mini`
  - docker config: `/workspace/configs/openaudio-s1-mini`
