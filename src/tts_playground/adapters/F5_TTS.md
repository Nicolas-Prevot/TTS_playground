# F5-TTS

## Voice Cloning Capability

- Zero-shot voice cloning: F5-TTS can closely mimic a new voice (including accent, intonation, and timbre) with just a few seconds of audio as reference (around 2–10 seconds).
- Suitable for generating speech in the style of a specific speaker or for replicating a target voice for assistive, entertainment, or accessibility tasks.

## Language Support

- Base F5-TTS supports English and Chinese out-of-the-box.
- Community models expand support to other languages such as French, Russian, Spanish, and Portuguese (Brazilian), via training and fine-tuning on new datasets.

## Model Size

- The typical model size is approximately 1.5GB for the full package (including all components).
- Main configuration (from research):
    - Parameters: Typically 150–160 million parameters (base model: 158M).
    - Training set: 95,000–100,000 hours of audio data.

## Architecture

- Fully non-autoregressive TTS model.
- Core technologies:
    - Diffusion Transformer (DiT): Enables high-quality synthesis by combining transformer and denoising diffusion frameworks.
    - Flow Matching: For more natural speech by guiding random noise into clear, expressive speech.
    - ConvNeXt blocks: Enhances text feature refinement before generation, improving alignment between text and speech.
    - Sway Sampling: Efficient, adaptive sampling strategy enabling quicker inference and high performance.

## Emotion Control

- Enables emotion-aware synthesis; can control output sentiment (happy, sad, angry, fearful, calm, etc.) by:
    - Uploading a reference audio sample with the desired emotion


|  | E2TTS_Small | E2TTS_Base | F5TTS_Base | F5TTS_Small | F5TTS_v1_Base |
| --- | --- | --- | --- | --- | --- |
| vocos | No | Yes | Yes | No | Yes |
| bigvgan | No | No | Yes | No | No |
