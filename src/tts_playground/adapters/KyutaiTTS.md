# Kyutai TTS (Kyutai 1.6B, English/French)

## Overview/Architecture

Kyutai TTS is a recently open-sourced real-time streaming TTS model. It is a Transformer-based architecture with separate text and audio token streams (a form of multimodal/dual-stream model) enabling it to start speaking before the full text is seen. The model has 1.6 billion parameters. It uses a custom “Mimi” audio codec to convert audio to discrete tokens, and the transformer generates audio tokens on-the-fly from incoming text tokens with only ~220 ms latency. Notably, Kyutai TTS was trained on a massive 2.5 million hours of audio (transcribed via Whisper). It is designed for real-time conversations – e.g., it powers Kyutai’s chat assistant “Moshi” with instantaneous TTS responses.

## Voice Cloning: No

Kyutai TTS can perform voice cloning from a ~10 second reference clip. It will mimic the speaker’s voice, intonation, and even recording characteristics (“mic quality”) in generated speech. However, to prevent misuse, the developers did not release the part of the model that allows arbitrary voice embedding generation. Instead, the open model only accepts voice embeddings from a preset collection of voices or newly donated voices. In practice, this means you can only clone voices that Kyutai provides (from public datasets) or you can contribute your own voice via their tool to get an embedding. Cloning random third-party voices is not directly enabled in the open release (as a security measure).

# Voices Available

Since free-form cloning is restricted, Kyutai offers a repository of pre-made voice embeddings for use. These include voices from datasets like VCTK (English accents), Expresso (conversational English with various emotions), CML-TTS (French voices), and EARS (English emotional speech, 107 speakers). In total, dozens of voices are provided (male and female, multiple accents). For example, the EARS set gives 107 distinct speaker voices, and even includes a couple of speakers recorded in multiple emotional tones (so you can use one speaker’s “angry” voice vs. “happy” voice by choosing different embeddings). Users can select any of these voices as the output speaker.

## Languages: English and French

Kyutai TTS is bilingual – it can speak fluent English or French, matching the text language. They are exploring adding more languages, but currently only EN and FR are supported by the model (the underlying LLM “Helium” supports 24 EU languages, but TTS outputs beyond EN/FR are not yet released). The model can maintain a cloned voice across both English and French (i.e., cross-lingual voice cloning) – for instance, clone an English speaker and have them speak French, or vice versa, given its bilingual training.

## Emotion Control: No

No direct user control via markers. Kyutai does not expose an API for arbitrarily changing emotion in speech on the fly. However, some preset voices include emotional styles as mentioned (e.g. a certain voice embedding might be an “angry female from dataset X”). So you can get different emotions by choosing an appropriately recorded voice from the library (one example: the EARS subset provides the same speaker in neutral vs. emotional recordings). But there is no parameter or text tag to make a given voice speak happier or sadder dynamically – it will speak in whatever style the reference voice inherently has. Kyutai’s focus is real-time natural delivery; it maintains consistent tone unless you swap the voice embedding.

## Model Size: 1.6 billion parameters

It’s a large model but optimized for streaming inference (the team achieved ~350 ms response serving 32 concurrent streams on a single GPU). The model uses BF16 precision and a highly optimized Rust server for deployment, making real-time performance feasible on high-end GPUs.

## Architecture Details

Dual-stream Transformer – one stream processes incoming text incrementally, and the other generates audio tokens with a slight delay (to allow a small lookahead). This “delayed streams modeling” is Kyutai’s key innovation, enabling true text-streaming TTS. The model is multimodal at its core (the underlying architecture “Helium” can do TTS, STT, QA, etc., though only TTS is released). It uses a custom Mimi codec (similar to Descript’s audio codec) to tokenize audio, and a transformer to predict these tokens. It was trained with RLHF fine-tuning to improve naturalness and handling of interactive speech (similar to how ChatGPT is tuned, but here applied to speech intonation). Additionally, Kyutai outputs precise word-level timestamps along with audio, which is very useful for aligning subtitles or enabling barge-in (the system knows exactly which word is being spoken at each moment).