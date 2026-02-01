# Kokoro Adapter (tts-adapter-kokoro)

Adapter for **Kokoro-82M** (the `kokoro` Python package) integrated into **TTS Playground**.

This adapter intentionally **does not do voice cloning from audio**. Instead, it selects a *pretrained* voice by ID
(e.g. `af_heart`, `bm_george`) or, optionally, a local voice embedding tensor (`.pt`).

---

## Requirements

### Python

- Python **>= 3.10** (upstream `kokoro` requires Python < 3.13, >= 3.10).

### System dependencies (important)

Kokoro uses **espeak-ng** for:
- **English out-of-dictionary fallback**, and
- **some non-English languages**.

If `espeak-ng` is missing, English may still work but OOD words can degrade, and some languages may fail.

**Debian/Ubuntu**
```bash
sudo apt-get update
sudo apt-get install -y espeak-ng libsndfile1
```

**Windows**
Follow the upstream “Windows Installation” notes from the `kokoro` PyPI page (espeak-ng MSI installer).

---

## Install (adapter venv)

```bash
cd adapters/kokoro
uv sync
```

### Optional language tokenizers (Misaki extras)

Upstream Kokoro notes that some languages require Misaki extras:

- Japanese: `misaki[ja]`
- Chinese: `misaki[zh]`

If you need them:

```bash
# Japanese
uv pip install "misaki[ja]"

# Chinese
uv pip install "misaki[zh]"
```

---

## Language codes & voice IDs

Upstream language codes:

- `a` => American English
- `b` => British English
- `e` => Spanish (es)
- `f` => French (fr-fr)
- `h` => Hindi (hi)
- `i` => Italian (it)
- `p` => Brazilian Portuguese (pt-br)
- `j` => Japanese (requires `misaki[ja]`)
- `z` => Mandarin Chinese (requires `misaki[zh]`)

**Important:** `lang_code` should match the voice you use. In most Kokoro voice IDs, the first character of the voice
matches the language code (e.g. `af_heart` => `a`, `bm_george` => `b`).

For the complete voice list, see the upstream `VOICES.md` in the `hexgrad/Kokoro-82M` model repo.

---

## Local Python usage

```python
from tts_adapter_kokoro.adapter import KokoroTTSAdapter

tts = KokoroTTSAdapter(lang_code="a", voice="af_heart")
tts.load_model()

# Switch voice (lang_code is optional; inferred from voice prefix if omitted)
tts.clone_voice(voice="bm_george", lang_code="b")

wav_bytes = tts.synthesize(
    "Empires rise and fall, but I remain.",
    speed=1.0,
    split_pattern=r"\n+",
)

with open("out.wav", "wb") as f:
    f.write(wav_bytes)
```

### Long text notes

Kokoro’s default split strategy is newline-based in many examples. For long passages:
- insert newlines in the text, **or**
- pass a `split_pattern` that fits your chunking needs.

---

## API usage (TTS Playground)

**POST** `http://localhost:7000/v1/tts`

```json
{
  "adapter": "kokoro",
  "init": { "lang_code": "a", "voice": "af_heart" },
  "load_model": {},
  "clone_voice": { "voice": "af_bella", "lang_code": "a" },
  "synthesize": { "text": "Hello world.", "kwargs": { "speed": 1.0 } }
}
```

---

## Troubleshooting

### 1) `espeak-ng` missing

Symptoms:
- warnings about missing espeak
- degraded pronunciation for uncommon English words
- failures for some non-English languages

Fix: install `espeak-ng` (see Requirements section).

### 2) Misaki / spaCy downloads inside uv env

Misaki (English) uses spaCy under the hood. If spaCy tries to download models at runtime,
ensure `pip` is present in the adapter venv and/or pre-download the required spaCy model
according to Misaki’s docs.

---
