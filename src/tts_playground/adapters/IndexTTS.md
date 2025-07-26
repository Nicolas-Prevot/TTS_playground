# Index-TTS

## Overview/Architecture

IndexTTS is a GPT-style TTS system combining ideas from XTTS and Tortoise-TTS. It uses a Transformer language model to generate discrete acoustic tokens, plus a Conformer encoder for the reference audio and a BigVGAN2 vocoder/decoder for waveform output. The architecture follows a hybrid codec approach: a VQVAE (dVAE) compresses audio into tokens, then a GPT-like model predicts those tokens from text (conditioned on speaker prompt). This yields high audio quality with faster inference than diffusion models. (It is optimized for industrial-level efficiency and controllability.)

## Usage Instructions

```bash
huggingface-cli download IndexTeam/IndexTTS-1.5 config.yaml bigvgan_discriminator.pth bigvgan_generator.pth bpe.model dvae.pth gpt.pth unigram_12000.vocab --local-dir checkpoints/indextts
conda env create -f environment_indextts.yml
conda activate index-tts
```

Example usage:
```python
tts = IndexTTSAdapter(
    model_dir="checkpoints/indextts",
    cfg_path="checkpoints/indextts/config.yaml",
    is_fp16=True,
    device="cuda",  # "cpu"
    use_cuda_kernel=True,  # set True if you built the custom CUDA ops
    fast=False,  # standard (higher-quality) inference
)

tts.load_model()

tts.clone_voice("data/gen/testkokoro.wav") # "data/ref/basic_ref_en.wav"

wav_bytes = tts.synthesize(
    "Hello, this is a demo of IndexTTS zero-shot voice cloning!",
    do_sample=True,           # sampling-based decoding
    top_p=0.9,                # nucleus sampling
    temperature=0.7,          # more conservative sampling
    num_beams=5,              # beam search width
    fast=False                # set True for faster but lower-quality output
)

with open("data/gen/index_output_demok.wav", "wb") as f:
    f.write(wav_bytes)
```

## Voice Cloning: Yes

Index-TTS supports zero-shot voice cloning. You can provide a short 5–10 second sample of a speaker’s voice, and the model will synthesize new speech in that voice. It achieves high speaker similarity (MOS ~4.2/5 for timbre) in evaluations, meaning it well-preserves the speaker’s vocal characteristics (accent, tone, etc.). The reference voice is passed via a --voice audio file in the CLI, and only a few seconds are sufficient to capture the voice’s essence.

# Languages: Bilingual

Trained on Chinese and English speech. It can synthesize in both Mandarin Chinese and English, and handle mixed-language input (Chinese characters with English words) with proper pronunciation. A special character-pinyin hybrid input method is supported for Chinese: users can input pinyin annotations to disambiguate character pronunciation. Cross-language voice cloning is possible (e.g. clone a Chinese speaker’s voice to speak English text).

## Emotion Control: No

No explicit emotion control. IndexTTS does not have emotion/style tokens or an emotion reference mechanism. Controllability in this model refers to other aspects: you can control pauses via punctuation (commas/periods to induce breaks), and ensure correct Chinese intonation via pinyin input. But there are no built-in emotional markers or prosody tags – the model’s style is largely driven by the reference voice’s natural speaking style and the punctuation in text.

## Model Size

The model uses a large Transformer for acoustic token generation – likely on the order of >300M parameters (given it runs in about 8GB GPU VRAM for v1.5). (It’s comparable in scale to other GPT-style TTS like Tortoise.) It was trained on “tens of thousands of hours” of speech data, indicating a robust model.

## Architecture Details

Autoregressive text-to-token generator with a Conformer conditioning module. The text is processed with a BPE tokenizer (no need for phonemes), and the model directly learns to map text to speech tokens. A BigVGAN2 vocoder then converts acoustic tokens to waveform, yielding high fidelity audio. Notably, IndexTTS introduced pinyin mixed input for Chinese (to handle polyphonic characters) and achieved nearly 100% VQ codebook usage by comparing VQ vs. Finite Scalar Quantization. Overall, it emphasizes stability and low latency: the v1.5 release improved English fluency and stability, with ~8GB VRAM usage and ~0.7–0.8 real-time speed on consumer GPUs.