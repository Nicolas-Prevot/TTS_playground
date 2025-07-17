# F5-TTS

## Overview/Architecture:

F5-TTS is a fully non-autoregressive diffusion-based TTS model using a Diffusion Transformer (DiT) with ConvNeXt2 backbone. It employs flow matching for speech generation (inspired by E2-TTS) without explicit duration or alignment models. This design yields fast training and inference (RTF ~0.15) and robust alignment via a filler-token padding approach. The base model has 335.8M parameters (22-layer Transformer + 4-layer ConvNeXt2).

## Voice Cloning: Yes

F5-TTS exhibits zero-shot voice cloning ability. It can mimic an unseen speaker’s timbre and prosody from a reference audio clip plus transcript. In practice ~15 seconds of reference audio yields high-quality cloning. (It was trained on a massive 100k hours multilingual dataset, which helps it generalize to new voices.) Accent and intonation of the reference speaker are retained in the generated speech.

## Languages: Multilingual

Trained on English and Chinese primarily (demonstrates seamless code-switching between them). It can handle mixed English–Chinese text input. Other languages are not explicitly mentioned as supported.

## Emotion Control: No explicit controls

F5-TTS focuses on faithful text-to-speech and natural prosody from data. There are no user-controllable emotion or style tokens in its interface (no mention of emotion tags or reference-based style mimicry in docs).

## Model Size: ~335 million parameters (base model)

Uses a pretrained vocoder (e.g. Vocos) to convert generated mel-spectrograms to waveform.

## Architecture Details

Fully non-autoregressive diffusion transformer model. It performs text-to-mel in a single stage via conditional flow-matching diffusion. ConvNeXt-V2 layers refine text embeddings for better alignment. No separate phonemizer or duration predictor is used – instead the text is padded to match audio length and denoising is done end-to-end. Note: F5-TTS requires an audio prompt (mel + transcript) at inference to define the target voice, and generates speech for new text in that voice.


|  | E2TTS_Small | E2TTS_Base | F5TTS_Base | F5TTS_Small | F5TTS_v1_Base |
| --- | --- | --- | --- | --- | --- |
| vocos | No | Yes | Yes | No | Yes |
| bigvgan | No | No | Yes | No | No |
