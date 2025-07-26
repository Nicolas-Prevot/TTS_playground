# OpenAudio S1-mini

## Overview/Architecture

OpenAudio S1-mini is a state-of-the-art open TTS model from FishAudio, representing a distilled version of their flagship S1 model. It uses a large-scale Transformer (LLM-based) architecture – specifically built on Qwen-3 (an LLM) adapted for TTS. The full S1 is 4B parameters (not publicly released) and S1-mini is 0.5B (500M) parameters. It is a multilingual, multi-speaker model with advanced capabilities. S1-mini was trained on >2 million hours of audio across 13 languages, and further fine-tuned with RLHF (reinforcement learning from human feedback) for natural voice quality. The model treats TTS as a sequence-to-sequence task with text and special tokens as input and audio tokens as output, leveraging an integrated audio codec (Descript-like) for speech generation.

## Usage Instructions

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
# copy fish_speech/configs/modded_dac_vq.yaml from https://github.com/fishaudio/fish-speech to configs/openaudio-s1-mini/modded_dac_vq.yaml
uv sync --extra fishaudio
```

Example usage:
```python
tts = OpenAudioS1MiniAdapter(
    llama_checkpoint_dir="checkpoints/openaudio-s1-mini",
    codec_checkpoint_path="checkpoints/openaudio-s1-mini/codec.pth",
    decoder_config_name="modded_dac_vq",
    device="cuda",      # or "cpu"
    half=False,         # use float32
)

tts.load_model()

tts.clone_voice("data/ref/basic_ref_en.wav",
                ref_text="Some call me nature, others call me mother nature.")

english_bytes = tts.synthesize(
    "(shouting)I don't really care what you call me. (shouting)I've been a silent spectator, "
    "watching species evolve, empires rise and fall. (shouting)But always remember, "
    "I am mighty and enduring, (laughing) Ha,ha,ha!",
    max_new_tokens=0,
    top_p=0.9,
    repetition_penalty=1.1,
    temperature=0.8
)

with open("data/gen/test_s1_eng2.wav", "wb") as f:
    f.write(english_bytes)
```

## Voice Cloning: Yes

OpenAudio S1 supports rapid zero-shot voice cloning. You can clone a voice by providing 30–45 seconds of reference audio (Best: 2-3 15-20s clips forming a complete paragraph). It will closely imitate the speaker’s voice, including timbre and speaking style. The cloning is very high-quality – the model’s training on diverse speakers and the RLHF tuning allow it to capture nuances of the reference voice. Notably, S1 can perform cross-lingual voice cloning: you can input a speaker’s sample in one language and have it speak text in another language while preserving the voice identity. (For example, clone an English speaker and have them speak Japanese.) In evaluations, OpenAudio’s voice similarity (“speaker distance”) is better than most open models. Cloning is done by feeding the reference audio to the model (the exact interface depends on the integration, e.g. through their UI or a prompt token).

# Languages: Multilingual (13 languages)

S1-mini supports English, Chinese, Japanese, German, French, Spanish, Korean, Arabic, Russian, Dutch, Italian, Polish, Portuguese. It can read text in any of these languages with correct pronunciation. It also handles multilingual text (e.g. an English sentence with a French phrase) and will produce appropriate pronunciation for each. The model was top-ranked on multilingual TTS benchmarks and even supports cross-lingual output using a single voice (as mentioned, you can maintain one speaker across different language texts).

## Emotion Control: Yes

Via rich text markers. OpenAudio S1 introduces an extensive set of inline emotion and style markers that users can put in the input text to control the speech’s emotional tone and voice effects. It covers 50+ emotions. For example, you can prefix a sentence with (angry) or (sad) to have it spoken in that emotion, or even more nuanced ones like (sarcastic), (empathetic), (furious), etc. Additionally, tone markers like (whispering) or (shouting) adjust the delivery style. There are also special effect markers such as (laughing), (sobbing), (sighing), even (crowd laughing) to insert background laughs. These markers are simply included in the input text, and the model will modify the speech accordingly. (For a complete list of supported markers, refer to the OpenAudio S1 documentation, which enumerates all emotions and effects.) This level of control is unprecedented in open TTS – S1 can literally laugh or cry on command and convey complex emotions through these tokens.

## Model Size: 500M parameters (for S1-mini)

The full S1 is 4B (used in FishAudio’s cloud service). Despite being 0.5B, S1-mini achieves nearly the same quality, thanks to distillation and RLHF. It runs in real-time on modern GPUs (reports indicate ~1:5 speed on an RTX 4060).

## Architecture Details

Transformer-based multimodal model. OpenAudio S1 leverages a large language model architecture (Qwen) but tailored for speech tasks. It is “native multimodal” – meaning the same model can potentially do speech-to-text, QA, etc., though only TTS is released publicly. The audio representation uses a descript-style codec (likely a vector-quantized codec) for high fidelity. The model is dual-mode (dual_ar): it likely uses an autoregressive generation for text→audio tokens but with mechanisms for real-time streaming (the details combine best of AR and non-AR for speed). S1-mini also underwent online RLHF training where human feedback tuned the model to produce more natural and context-appropriate prosody. This fine-tuning, plus the comprehensive marker system, gives it unprecedented control and quality. In tests, OpenAudio S1-mini achieves extremely low WER (~0.01) and was ranked #1 on the TTS-Arena benchmark, even surpassing many closed-source systems. It is considered a new state-of-the-art in open TTS, combining high-quality cloning, emotion versatility, and multilingual prowess in one package.

    - Basic emotions:
    (angry) (sad) (excited) (surprised) (satisfied) (delighted) 
    (scared) (worried) (upset) (nervous) (frustrated) (depressed)
    (empathetic) (embarrassed) (disgusted) (moved) (proud) (relaxed)
    (grateful) (confident) (interested) (curious) (confused) (joyful)

    - Advanced emotions:
    (disdainful) (unhappy) (anxious) (hysterical) (indifferent) 
    (impatient) (guilty) (scornful) (panicked) (furious) (reluctant)
    (keen) (disapproving) (negative) (denying) (astonished) (serious)
    (sarcastic) (conciliative) (comforting) (sincere) (sneering)
    (hesitating) (yielding) (painful) (awkward) (amused)

    - Tone markers:
    (in a hurry tone) (shouting) (screaming) (whispering) (soft tone)

    - Special audio effects:
    (laughing) (chuckling) (sobbing) (crying loudly) (sighing) (panting)
    (groaning) (crowd laughing) (background laughter) (audience laughing)