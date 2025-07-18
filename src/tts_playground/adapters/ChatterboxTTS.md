# ChatterboxTTS

## Overview/Architecture

Chatterbox is a transformer-based TTS model from Resemble AI, built on a 0.5B-parameter LLaMA decoder backbone. It’s autoregressive, producing speech from text (and optional audio prompt) with high stability due to alignment-informed inference. A HiFi-GAN variant (HiFT-GAN) is integrated as the vocoder. Outputs are watermarked at a perceptual level for ethical safeguards. The model was trained on 500k hours of cleaned speech, enabling very natural results.

## Usage Instructions

```bash
uv sync --extra chatterbox
```

Example usage:
```python
tts = ChatterboxTTSAdapter()

tts.load_model()

tts.clone_voice("data/ref/basic_ref_en.wav")

audio_bytes = tts.synthesize(
    "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring.",
    cfg_weight=0.5,
    exaggeration=0.5,
)

with open("data/gen/output_classic.wav", "wb") as f:
    f.write(audio_bytes)
```

## Voice Cloning: Yes

Chatterbox supports zero-shot voice cloning. It can clone any voice from ~5 seconds of reference audio, capturing the speaker’s timbre, accent and intonation. The user provides a short audio sample (“audio_prompt”) and the model will generate new speech in that voice. This model was explicitly designed for high-quality voice mimicry and is often preferred over ElevenLabs in blind tests.

## Languages: English-only

Currently Chatterbox is trained for English synthesis. (No multilingual support as of its initial release.)

## Emotion Control: Yes
It is the first open-source TTS to offer adjustable emotion intensity. There is a continuous “exaggeration” parameter to control expressiveness. At exaggeration=0.0 the delivery is flat/monotone, while higher values (e.g. 0.7) produce more dramatic, emotive speech. This is not done via discrete emotion tokens, but via this tuning knob in the API. (There are no special text markers; emotion is controlled programmatically by setting the parameter.) Additionally, a “cfg_weight” parameter controls how closely to follow the reference voice’s style vs. neutral reading. Together these allow nuanced emotional expression (e.g. more expressive, slower, etc.).

## Model Size: ~500 million parameters

The core is a 0.5B LLaMA-derived transformer model. Despite its size, it runs in ~6.5GB VRAM and can do real-time or faster-than-real-time inference on a GPU.

## Architecture Details

Decoder-only Transformer architecture (inspired by LLaMA) for text-to-speech. It operates in a two-stage approach: an acoustic model (text → mel or intermediate representation) plus a vocoder (mel → waveform). Chatterbox uses alignment models to ensure stable pronunciation (no skipped or repeated words), and it inserts an imperceptible digital watermark into all outputs to detect. Training data included a wide variety of speakers and styles, enabling strong zero-shot generalization. (Licensed under MIT; model weights CC-BY-NC.)