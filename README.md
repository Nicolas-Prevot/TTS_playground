# TTS Playground

This project is still **ongoing**, feedback is appreciated if you spot an issue, have suggestions / recommendations.

## Introduction

**TTS Playground** is a project designed to make it easy to evaluate and **compare several advanced TTS models** side-by-side. It provides a working environment (or “adapter”) for each included model so you can load them, perform **voice cloning** (where supported), and synthesize speech with a unified interface. The project is towards anyone interested in **comparing top open-source TTS systems** in order to select the right one for their needs.

Interactive page with audio samples: **[🎧 Samples Page](https://Nicolas-Prevot.github.io/TTS_playground/tts_samples.html)**

Key features of TTS Playground include:

* **Multiple integrated models:** Test and switch between different TTS models easily within one codebase. Currently **7 models** are supported (listed below), covering a range of architectures and capabilities (diffusion vs. autoregressive, multilingual, etc.).
* **Unified interface:** A common API to clone a voice and synthesize speech, regardless of the underlying model. This makes it straightforward to feed the same input to each model and compare outputs.
* **Voice cloning and multi-speaker support:** Most of the integrated models support **zero-shot voice cloning** – providing a short reference audio of a speaker allows the model to generate speech in that voice. This is useful for testing how well each model mimics an unseen speaker.
* **Multilingual and expressive speech:** Several models support multiple languages (English, Chinese, French, etc.) and some offer **emotional tone control** (either via special input markers or model parameters).
* **Audio sample comparisons:** We provide an interactive **[🎧 Samples Page](https://Nicolas-Prevot.github.io/TTS_playground/tts_samples.html)** that demonstrates the output of each model on the same inputs. You can listen to side-by-side audio clips for voice cloning in English and French, as well as hear the preset voices from models that don’t clone arbitrarily (e.g. Kokoro82M, Kyutai). This helps in qualitatively comparing the naturalness, expressiveness, and speaker similarity across models.

## Supported Models

**TTS Playground currently integrates 7 TTS models:**

* [**F5-TTS**](https://github.com/SWivid/F5-TTS) – a non-autoregressive diffusion-based model (335M params) with flow-matching, supporting **zero-shot voice cloning** (needs \~15 seconds of reference audio). It’s primarily trained on English and Chinese and focuses on fast, high-quality speech generation using a ConvNeXt2 + Transformer architecture.
* [**ChatterboxTTS**](https://github.com/resemble-ai/chatterbox) – an autoregressive transformer model (\~500M params) from Resemble AI built on a LLaMA 0.5B backbone. It supports cloning from \~5 seconds of audio and is **English-only**. Notably, it offers an *“exaggeration”* parameter to control emotion intensity in the output (making it one of the first open TTS with adjustable expressiveness).
* [**Index-TTS**](https://github.com/index-tts/index-tts) – a GPT-style text-to-speech system (\~300–400M params) that uses a transformer to predict discrete audio tokens (inspired by models like Tortoise-TTS). It supports **voice cloning** from 5–10 seconds of audio and is bilingual (trained on English and Chinese, with cross-language cloning capability). This model focuses on efficient generation with high fidelity, using a Conformer for the reference encoder and a BigVGAN2 vocoder.
* [**Kokoro 82M**](https://github.com/hexgrad/kokoro) – an ultra-lightweight (82M params) multi-speaker TTS model that does **not support arbitrary voice cloning**. Instead, Kokoro comes with a set of **pre-trained voice embeddings** (dozens of voices across languages) that you can choose from. It supports many languages (English, Japanese, Chinese, French, Spanish, Hindi, etc.) via these preset voices. Kokoro is based on a StyleTTS2 architecture (non-AR) with an iSTFTNet vocoder, emphasizing efficiency (real-time synthesis even on CPU). *(See the Kokoro project’s [Voice List](https://github.com/RVC-Project/Kokoro) for available voices.)*
* [**Kyutai TTS**](https://kyutai.org/next/tts) – a large bilingual (English/French) TTS that offers **streaming real-time generation**. Officially, the open-source release does *not allow free-form voice cloning* – it can only use **preset or user-donated voices** rather than arbitrary new voices. (In practice, it can mimic a voice from \~10 seconds of audio, but only if that voice’s embedding is provided by the Kyutai system.) It does not have external emotion controls. The model uses a **dual-stream Transformer** architecture (text and audio tokens processed in parallel) optimized for low latency, and was trained on an enormous 2.5M hours of speech, enabling very natural prosody. *(See the Kyutai [Voices](https://huggingface.co/Kyutai) repository for available voice embeddings. Kyutai’s open model requires using one of these provided voices for output.)*
* [**OpenAudio S1-mini**](https://github.com/fishaudio/fish-speech) – a 500M-parameter **multilingual** TTS model released by FishAudio, distilled from their 4B-param S1 model. It supports **zero-shot voice cloning** (typically needs 30–45 seconds of reference audio; cross-lingual cloning is possible) and can speak **13 languages** (English, Chinese, Japanese, French, Spanish, and more). A standout feature is its **extensive emotion control via inline text markers** – over 50 expressive tags like `(angry)`, `(laughing)`, `(whispering)`, etc., to modulate the speech output. S1-mini uses a Transformer (based on Qwen-3 LLM) for sequence-to-sequence generation with an integrated audio codec, and has been fine-tuned with RLHF for natural voice quality. *(See FishAudio’s [documentation](https://github.com/FishAudio/OpenAudio-S1) for the list of supported markers and usage details.)*
* [**HiggsAudio v2**](https://github.com/boson-ai/higgs-audio) – a cutting-edge large TTS model from Boson AI (**\~5.8B parameters** combined) that merges an LLM (Llama 3B) with a specialized audio extension (Dual-FFN, \~2.2B). Higgs Audio supports **zero-shot voice cloning** (and even one-shot fine-tuning), achieving extremely high-quality voice mimicry. It is **multilingual** and was trained on a massive 10+ million hours of audio data, giving it broad language understanding and expressive output. While it doesn’t use explicit emotion tokens, it **excels at expressive speech generation** – automatically adjusting prosody and even generating multi-speaker dialogues with different voices. This model is at the state-of-the-art, outperforming many others (even some proprietary systems) in voice quality and expressiveness. *(Higgs Audio V2 is open-source; see BosonAI’s GitHub for technical details.)*

Below is a **feature comparison table** summarizing the above models:

| **Model**             | **Voice Cloning** | **Clone Details** (min. reference)     | **Prebuilt Voices**                                          | **Languages**                | **Emotion Control**      | **Model Size**                                                               |
| --------------------- | ----------------- | -------------------------------------- | ------------------------------------------------------------ | ---------------------------- | ------------------------ | -------------- |
| [**F5-TTS**](https://github.com/SWivid/F5-TTS)            | ✅ Yes             | \~15 s audio, clones timbre & prosody  | N/A                                                          | English, Chinese             | ❌ No                     | \~335M         | Non-AR diffusion (ConvNeXt2 + Transformer); flow-matching training; vocoder: Vocos. Fast inference (RTF \~0.15).                                                                  |
| [**ChatterboxTTS**](https://github.com/resemble-ai/chatterbox)     | ✅ Yes             | \~5 s audio, clones timbre & accent    | N/A                                                          | English-only                 | ✅ *Exaggeration* param   | \~500M         | Autoregressive Transformer (LLaMA 0.5B backbone); HiFi-GAN vocoder; alignment-informed decoding for stable output. Watermarked outputs for safety.                                |
| [**Index-TTS**](https://github.com/index-tts/index-tts)         | ✅ Yes             | \~5–10 s audio, clones timbre & accent | N/A                                                          | English, Chinese             | ❌ No                     | \~300M est.    | GPT-style autoregressive generation; Conformer reference encoder; BigVGAN2 vocoder. High-fidelity codec-based TTS optimized for speed.                                            |
| [**Kokoro 82M**](https://github.com/hexgrad/kokoro)        | ❌ No              | *(N/A – no arbitrary cloning)*         | ✔️ [Voice List](https://github.com/RVC-Project/Kokoro)       | Multi (EN, JP, CN, FR, etc.) | ❌ No                     | 82M            | StyleTTS2-based non-AR model + iSTFTNet vocoder; uses fixed learned speaker embeddings (select from dozens of provided voices). Ultra-fast and lightweight.                       |
| [**Kyutai TTS**](https://kyutai.org/next/tts)        | ❌ No\*            | \~10 s audio (restricted cloning)\*\*  | ✔️ [Voices](https://huggingface.co/Kyutai)                   | English, French              | ❌ No                     | 1.6B           | Dual-stream Transformer architecture; real-time streaming TTS (low latency \~220ms); uses “Mimi” audio codec. Trained on 2.5M hours Whisper-transcribed data for natural prosody. |
| [**OpenAudio S1-mini**](https://github.com/fishaudio/fish-speech) | ✅ Yes             | \~30–45 s audio, clones voice & style  | ✔️ [Markers/Docs](https://github.com/FishAudio/OpenAudio-S1) | 13 languages                 | ✅ Yes (50+ text markers) | 500M           | Transformer (Qwen LLM-based) seq2seq model; integrated Descript-style audio codec; **RLHF**-tuned for naturalness. SOTA multilingual and emotional expressiveness.                |
| [**HiggsAudio v2**](https://github.com/boson-ai/higgs-audio)     | ✅ Yes             | \~5–10 s audio, high-fidelity cloning  | N/A                                                          | English, Chinese (*multi*）   |  ✅❌No\*\*\*               | \~5.8B         | Large LLM-TTS hybrid (Llama 3B + 2.2B audio adapter); supports multi-speaker dialogues, automatic emotion/prosody adaptation. Trained on 10M hours for SOTA quality.              |

**\* Note:** *Kyutai’s open-source model does not allow arbitrary third-party voice cloning – it can only use preset or contributed voices. “No” indicates you cannot freely clone any voice with Kyutai (without special access).*

**\*** *HiggsAudio is **highly expressive**, but it does not have user-facing emotion control tokens or parameters – instead, it adjusts prosody and emotion automatically from context. In practice it can convey emotion extremely well, but there’s no explicit knob or marker for the user to set.*

Each model’s name above links to more information (if available) such as voice lists or documentation. In addition, the repository contains detailed write-ups for each model under `src/tts_playground/adapters/` (one Markdown file per model). Those provide in-depth **architectural descriptions, usage instructions, and specific tips** for each TTS system. If you are interested in a particular model’s inner workings or setup, be sure to read its adapter README.

## Audio Examples

An extensive **audio sample comparison** is available on the **[🎧 Samples Page](https://Nicolas-Prevot.github.io/TTS_playground/tts_samples.html)** (hosted via GitHub Pages). This page includes:

* **Voice cloning demos in English and French:** multiple reference voice clips and each model clones the voice and speak the same lines. You can play the original reference audio and each model’s output to hear how well they capture the speaker’s identity.
* **Preset voice showcases for Kokoro82M and KyutaiTTS:** since those models rely on built-in voices, a selection of those voices speaking a common sentence is used. This demonstrates the range of voices available in those models.

## Installation & Setup

*Tested on windows only*

**Each model has its own README, check them out for detailled Installation & Setup**.

**Recommended: Using `uv`:** If you have `uv` installed, you can simply run `uv venv --seed` in the project directory to create a virtual environment and install all base dependencies. Then, for each model you want to use, run `uv sync --extra <model>` to install that model’s additional requirements (e.g. `uv sync --extra f5` for F5-TTS). This will automatically handle pulling the correct packages (including from custom indexes or git if needed). *(Note: Index-TTS is not included in the above because it requires a special setup – see **IndexTTS adapter README**.)*

## Usage

You can use TTS Playground in two main ways: via a **Python API** (importing and calling the adapter classes in code), or via a **command-line interface (CLI)** for quick testing. Below are examples of each.

**1. Programmatic Usage (Python):** Each model has an adapter class (e.g., `F5TTSAdapter`, `ChatterboxTTSAdapter`, etc.) that exposes a simple interface: `load_model()`, `clone_voice(...)`, and `synthesize(...)`. You can import these from the `tts_playground.adapters` module. For example, to use the F5-TTS model in a Python script or notebook:

```python
from tts_playground.adapters import F5TTSAdapter

# Initialize the TTS adapter (you can specify device or other params if needed)
tts = F5TTSAdapter(model_name="F5TTS_v1_Base", vocoder_name="vocos")

tts.load_model()  # load the model weights

# Clone a voice from a reference audio (and corresponding transcript, if required)
tts.clone_voice("path/to/speaker.wav", ref_text="Transcript of speaker.wav goes here")

# Now synthesize speech from text using the cloned voice
audio_bytes = tts.synthesize("Hello, this is a test of the F5-TTS model.")

# Save the audio to a file
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

All adapters follow a similar pattern, but the exact parameters can vary by model. For instance, some models don’t require a `ref_text` for cloning (just audio), some have extra options like `exaggeration` (Chatterbox) or `speed` (Kokoro) in `synthesize()`. Refer to each model’s README for the specific usage examples – they show the expected parameters and any special behavior (for example, Chatterbox’s `cfg_weight` and `exaggeration` controls, or OpenAudio’s use of text markers for emotions).

**2. CLI Usage:** For convenience, the project provides a unified inference script `inference_demo.py` that you can run from the command line. This allows you to generate an audio sample with any model by specifying a few arguments. For example:

```bash
# Example: use ChatterboxTTS to clone a voice and generate speech
python src/tts_playground/inference_demo.py --model "ChatterboxTTS" \
    --ref_audio data/ref/en/belinda.wav \
    --text "Hello, this is a voice cloning test of Chatterbox." \
    --out out_chatterbox.wav
```

In the above, `--model` chooses the model, `--ref_audio` is a path to a reference .wav file for cloning (if the model supports cloning; you can omit this for models that don’t clone or if you want default voice), `--text` is the input text to speak, and `--out` is the output filename for the synthesized speech. Some models also accept `--ref_text` if they require the transcript of the reference audio (e.g. F5-TTS and OpenAudio S1-mini use this to better clone style). The script will take care of loading the model and producing the output wave file.

This CLI is handy for quickly trying each model from the terminal. Under the hood, it uses the same adapter classes. You can also pass JSON strings to override any model-specific settings (see `--clone_args`, `--synth_args` in the script usage) if you want to tweak parameters like temperature, etc., without writing code.

## Next Steps and Contribution

This project is **ongoing** and welcomes contributions or suggestions. Planned improvements include refining the installation process (perhaps providing a one-click script or Docker), adding more models as they become available, and improving the sample page (e.g., more examples or a nicer UI). If you have a model you’d like to see integrated, or if you have sample prompts that reveal interesting differences between models, please contribute!

Feedback is also appreciated – if you discover that one model performs significantly better on a certain kind of text or notice any errors in the adapter implementations, let me know via GitHub issues. The goal is to make TTS Playground a comprehensive and up-to-date tested for the TTS community.

---


TODO:
- make checkpoints and artifacts automatically downloaded
- upgrade Interactive page with audio samples

Model to add:
- https://huggingface.co/maya-research/maya1
- https://huggingface.co/stepfun-ai/Step-Audio-EditX