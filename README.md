# TTS Playground

 uv venv --seed

 [🎧 Samples Page](https://Nicolas-Prevot.github.io/TTS_playground/tts_samples.html)


| Model           | Voice Cloning | Clone Details (Seconds/Features)                  | Prebuilt Voices / Link                      | Languages                  | Emotion Control                    | Model Size    | Architecture Overview                                                                                  |
|-----------------|----------------|--------------------------------------------------|---------------------------------------------|----------------------------|------------------------------------|----------------|---------------------------------------------------------------------------------------------------------|
| **F5-TTS**      | ✅ Yes          | ~15s, clones timbre, accent, intonation          | N/A                                         | English, Chinese           | ❌ No                              | ~335M          | Non-autoregressive diffusion (ConvNeXt2 + Transformer), flow matching, vocoder: Vocos                   |
| **ChatterboxTTS** | ✅ Yes        | ~5s, clones timbre, accent, intonation           | N/A                                         | English only               | ✅ Exaggeration parameter    | ~500M          | Autoregressive transformer (LLaMA 0.5B), HiFi-GAN vocoder, alignment-informed inference                 |
| **Index-TTS**   | ✅ Yes          | ~5-10s, clones timbre, accent, intonation        | N/A                                         | English, Chinese           | ❌ No                              | ~300-400M est. | Autoregressive GPT-style, Conformer encoder, BigVGAN2 vocoder, token-based generation                   |
| **Kokoro 82M**  | ❌ No           | N/A                                              | ✔️ [Voice List](https://github.com/RVC-Project/Kokoro) | Multi (EN, CN, JP, FR, etc.) | ❌ No                              | 82M            | StyleTTS2-based, non-AR transformer + iSTFTNet vocoder, fixed learned speaker embeddings                |
| **Kyutai fr**   | ❌ No*         | ~10s, preset voices or donation (restricted cloning) | ✔️ [Voices](https://huggingface.co/Kyutai) | English, French            | ❌ No | 1.6B           | Dual-stream Transformer, streaming real-time TTS, Mimi codec, optimized for low-latency conversational use |
| **OpenAudio S1 mini** | ✅ Yes     | ~10-45s, clones timbre, style, cross-lingual     | ✔️ [Markers/Docs](https://github.com/FishAudio/OpenAudio-S1) | 13 languages               | ✅ 50+ inline markers (e.g., (angry), (laughing)) | 500M           | Transformer (Qwen-based), sequence-to-sequence with text/audio tokens, RLHF tuned, multilingual          |
