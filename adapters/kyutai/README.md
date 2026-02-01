# Kyutai (Moshi) TTS Adapter

Adapter for **Kyutai TTS 1.6B** (`kyutai/tts-1.6b-en_fr`) using the **Moshi** PyTorch package.

This model is **English + French** and generates audio through the **Mimi** neural audio codec.

---

## Important limitations (voice selection)

This adapter **does not do arbitrary zero-shot voice cloning**.

You must select a **voice ID that exists in the Kyutai voice bank** (the Hugging Face repo `kyutai/tts-voices`). The voice IDs look like:

- `vctk/p225_023.wav`
- `cml-tts/fr/4724_3731_000031-0001.wav`
- `expresso/ex01-ex02_default_001_channel1_168s.wav`

These `.wav` files are present in `kyutai/tts-voices` alongside per-voice embedding files (e.g. `.safetensors`).

---

## Installation (adapter-only)

From the repository root:

```bash
cd adapters/kyutai
uv sync
```

Notes:

- `moshi` requires Python **>= 3.10** and is typically used with PyTorch 2.2+; a GPU is recommended for Moshi/Mimi workloads.
- If you hit `torch.compile()` / dynamo issues in your environment, Moshi documents disabling compilation with `NO_TORCH_COMPILE=1`.

---

## Local usage (direct Python)

```python
from tts_adapter_kyutai import KyutaiTTSAdapter

tts = KyutaiTTSAdapter(device="cuda", temp=0.6, cfg_coef=1.0)
tts.load_model()

# List known voices (IDs you can pass to clone_voice)
print(tts.model.get_voice_names()[:10])

tts.clone_voice("vctk/p225_023.wav")
wav_bytes = tts.synthesize("Hello from Kyutai TTS!")
open("out.wav", "wb").write(wav_bytes)
```

### Parameters

| Parameter | Where it applies | Meaning |
|---|---:|---|
| `temp` | **init** | sampling temperature used by the model |
| `n_q` | **init** | number of Mimi codebooks / quantizers |
| `cfg_coef` | init **or** per-call | conditioning strength (Moshi notes this is from **CFG distillation**, not classic CFG at inference) |

Per-call override example:

```python
wav_bytes = tts.synthesize("More controlled.", cfg_coef=2.0)
```

---

## API usage (TTS Playground)

**POST** `http://localhost:7000/v1/tts`

```json
{
  "adapter": "kyutai",
  "init": { "device": "cuda", "temp": 0.6, "cfg_coef": 1.0 },
  "load_model": {},
  "clone_voice": { "voice_sample": "vctk/p225_023.wav" },
  "synthesize": { "text": "Bonjour, comment allez-vous ?", "kwargs": {} }
}
```

If you want to override `cfg_coef` *per request* without re-initializing:

```json
{
  "adapter": "kyutai",
  "init": { "device": "cuda", "temp": 0.6, "cfg_coef": 1.0 },
  "load_model": {},
  "clone_voice": { "voice_sample": "vctk/p225_023.wav" },
  "synthesize": { "text": "Test.", "kwargs": { "cfg_coef": 2.0 } }
}
```

---

## Troubleshooting

### “Unknown voice id …”
Use:

```python
tts.load_model()
print(tts.model.get_voice_names()[:50])
```

Pick one of those IDs, or browse `kyutai/tts-voices` to find available voices. 

### Torch compile / dynamo issues
Try:

```bash
export NO_TORCH_COMPILE=1
```

(as documented by Moshi).
