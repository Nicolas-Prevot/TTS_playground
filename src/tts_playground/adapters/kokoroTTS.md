# Kokoro 82M

## Overview/Architecture

Kokoro is an ultra-lightweight TTS model (only 82 million parameters total). It follows the StyleTTS 2 architecture (non-autoregressive acoustic model) with an iSTFTNet neural vocoder. The model is multilingual and multi-speaker, but does not use zero-shot cloning. Instead, Kokoro comes with a set of learned voice embeddings (“voice tokens”) for a library of speakers. At runtime you pick one of the provided voices. This design keeps the model small and efficient – it can generate speech faster than real time even on modest hardware. Training was done on <1000 hours of permissively licensed data, making Kokoro inexpensive to train (≈$1k on A100 GPUs).

```bash
uv sync --extra kokoro
```

Example usage:
```python
tts = KokoroTTSAdapter()

tts.load_model()

tts.clone_voice(lang_code="a", voice="af_bella")

audio_bytes = tts.synthesize("I don't really care what you call me. I've been a silentspectator, watching species evolve, empires rise and fall. But always remember, I am mightyand enduring.", speed=1.0)

with open("data/gen/testkokoro.wav", "wb") as f:
    f.write(audio_bytes)
```

## Voice Cloning: No

Kokoro does not support arbitrary voice cloning from a new sample. You cannot input a random speaker’s voice; instead you choose from the built-in voices. Each voice is represented by a learned embedding vector. These voices were “curated and effective” and cover a variety of genders, accents, and languages. (The model’s focus is efficiency and simplicity, foregoing zero-shot cloning to keep size small.)

## Voices Available

Kokoro provides a library of voices (each corresponding to a particular speaker style). There are dozens of voices across multiple languages: e.g. 20 English voices (American/British, male and female), 5 Japanese voices, 8 Chinese (Mandarin) voices, 3 Spanish, 1 French, 4 Hindi, etc. (See the official voice list for full details.) These voice embeddings were derived from datasets like LibriTTS/VCTK and others; each voice has an ID (like af_heart for an American female voice). The user selects a voice ID which conditions the synthesis pipeline.

## Languages: Multilingual

Kokoro supports speech generation in a wide range of languages, including English (US/UK), Japanese, Mandarin Chinese, Spanish, French, Hindi, Italian, Brazilian Portuguese, etc.. In total it has voices (and corresponding text frontend/G2P) for at least ~9 languages. Note: The model relies on an internal G2P (grapheme-to-phoneme) library and espeak for pronunciation in each language. Some languages with limited training data (e.g. French with only 1 voice) may have lower quality or require careful input.

## Emotion Control: No

Kokoro’s voices generally have a neutral speaking style. There is no explicit mechanism to control emotion or speaking style via markers or reference audio. All generated speech tends to be neutral/professional-sounding (users noted it can be somewhat monotonous). The only control is implicit: selecting a different voice ID might yield a different tone (some voices sound more expressive or have slight emotion if their training data did). But no built-in tags for emotions like “happy” or “sad” are supported.

## Model Size: 82M parameters

Extremely small for a TTS model. This includes both the acoustic model and vocoder. Kokoro is optimized for speed and low resource usage; it can run on CPU or small GPUs easily. Despite its size, its quality is comparable to much larger models for normal-length sentences.

## Architecture Details

Based on StyleTTS2 (a non-autoregressive transformer that generates spectrograms from text and a style embedding) and ISTFTNet vocoder. It uses no diffusion, no autoregression – it generates the speech in one pass. The “style” is provided by the fixed voice embeddings for each speaker. Training involved IPA phoneme inputs for robust pronunciation. Kokoro handles punctuation and pauses well, and supports controlling speech speed via an optional parameter. It does not require an external alignment model; it learned to produce aligned speech from scratch (enabled by plenty of clean, short audio clips in training). Overall, Kokoro demonstrates that a well-designed lightweight model can achieve competitive naturalness with far less compute.


## 🇺🇸 American English  
`lang_code='a'` in `misaki["en"]`  
Fallback: `espeak-ng en-us`

| Name        | Traits     | Target Quality | Training Duration | Grade | SHA256     |
|-------------|------------|----------------|-------------------|--------|-------------|
| af_heart    | 🚺❤️        |                |                   | A      | 0ab5709b    |
| af_alloy    | 🚺         | B              | MM minutes        | C      | 6d877149    |
| af_aoede    | 🚺         | B              | H hours           | C+     | c03bd1a4    |
| af_bella    | 🚺🔥        | A              | HH hours          | A-     | 8cb64e02    |
| af_jessica  | 🚺         | C              | MM minutes        | D      | cdfdccb8    |
| af_kore     | 🚺         | B              | H hours           | C+     | 8bfbc512    |
| af_nicole   | 🚺🎧        | B              | HH hours          | B-     | c5561808    |
| af_nova     | 🚺         | B              | MM minutes        | C      | e0233676    |
| af_river    | 🚺         | C              | MM minutes        | D      | e149459b    |
| af_sarah    | 🚺         | B              | H hours           | C+     | 49bd364e    |
| af_sky      | 🚺         | B              | M minutes 🤏      | C-     | c799548a    |
| am_adam     | 🚹         | D              | H hours           | F+     | ced7e284    |
| am_echo     | 🚹         | C              | MM minutes        | D      | 8bcfdc85    |
| am_eric     | 🚹         | C              | MM minutes        | D      | ada66f0e    |
| am_fenrir   | 🚹         | B              | H hours           | C+     | 98e507ec    |
| am_liam     | 🚹         | C              | MM minutes        | D      | c8255075    |
| am_michael  | 🚹         | B              | H hours           | C+     | 9a443b79    |
| am_onyx     | 🚹         | C              | MM minutes        | D      | e8452be1    |
| am_puck     | 🚹         | B              | H hours           | C+     | dd1d8973    |
| am_santa    | 🚹         | C              | M minutes 🤏      | D-     | 7f2f7582    |

---

## 🇬🇧 British English  
`lang_code='b'` in `misaki["en"]`  
Fallback: `espeak-ng en-gb`

| Name        | Traits     | Target Quality | Training Duration | Grade | SHA256     |
|-------------|------------|----------------|-------------------|--------|-------------|
| bf_alice    | 🚺         | C              | MM minutes        | D      | d292651b    |
| bf_emma     | 🚺         | B              | HH hours          | B-     | d0a423de    |
| bf_isabella | 🚺         | B              | MM minutes        | C      | cdd4c370    |
| bf_lily     | 🚺         | C              | MM minutes        | D      | 6e09c2e4    |
| bm_daniel   | 🚹         | C              | MM minutes        | D      | fc3fce4e    |
| bm_fable    | 🚹         | B              | MM minutes        | C      | d44935f3    |
| bm_george   | 🚹         | B              | MM minutes        | C      | f1bc8122    |
| bm_lewis    | 🚹         | C              | H hours           | D+     | b5204750    |

---

## 🇯🇵 Japanese  
`lang_code='j'` in `misaki["ja"]`  
Total training data: `H hours`

| Name           | Traits     | Target Quality | Duration       | Grade | SHA256     | CC BY License     |
|----------------|------------|----------------|----------------|--------|-------------|--------------------|
| jf_alpha       | 🚺         | B              | H hours        | C+     | 1bf4c9dc    |                    |
| jf_gongitsune  | 🚺         | B              | MM minutes     | C      | 1b171917    | gongitsune         |
| jf_nezumi      | 🚺         | B              | M minutes 🤏   | C-     | d83f007a    | nezuminoyomeiri    |
| jf_tebukuro    | 🚺         | B              | MM minutes     | C      | 0d691790    | tebukurowokaini    |
| jm_kumo        | 🚹         | B              | M minutes 🤏   | C-     | 98340afd    | kumonoito          |

---

## 🇨🇳 Mandarin Chinese  
`lang_code='z'` in `misaki["zh"]`  
Total training data: `H hours`

| Name         | Traits     | Target Quality | Duration       | Grade | SHA256     |
|--------------|------------|----------------|----------------|--------|-------------|
| zf_xiaobei   | 🚺         | C              | MM minutes     | D      | 9b76be63    |
| zf_xiaoni    | 🚺         | C              | MM minutes     | D      | 95b49f16    |
| zf_xiaoxiao  | 🚺         | C              | MM minutes     | D      | cfaf6f2d    |
| zf_xiaoyi    | 🚺         | C              | MM minutes     | D      | b5235dba    |
| zm_yunjian   | 🚹         | C              | MM minutes     | D      | 76cbf8ba    |
| zm_yunxi     | 🚹         | C              | MM minutes     | D      | dbe6e1ce    |
| zm_yunxia    | 🚹         | C              | MM minutes     | D      | bb2b03b0    |
| zm_yunyang   | 🚹         | C              | MM minutes     | D      | 5238ac22    |

---

## 🇪🇸 Spanish  
`lang_code='e'` in `misaki["en"]`  
Fallback: `espeak-ng es`

| Name      | Traits   | SHA256     |
|-----------|----------|-------------|
| ef_dora   | 🚺       | d9d69b0f    |
| em_alex   | 🚹       | 5eac53f7    |
| em_santa  | 🚹       | aa8620cb    |

---

## 🇫🇷 French  
`lang_code='f'` in `misaki["en"]`  
Fallback: `espeak-ng fr-fr`  
Total training data: `<11 hours`

| Name      | Traits   | Target Quality | Duration     | Grade | SHA256     | CC BY |
|-----------|----------|----------------|--------------|--------|-------------|--------|
| ff_siwis  | 🚺       | B              | <11 hours    | B-     | 8073bf2d    | SIWIS  |

---

## 🇮🇳 Hindi  
`lang_code='h'` in `misaki["en"]`  
Fallback: `espeak-ng hi`  
Training duration: `H hours`

| Name      | Traits   | Target Quality | Duration     | Grade | SHA256     |
|-----------|----------|----------------|--------------|--------|-------------|
| hf_alpha  | 🚺       | B              | MM minutes   | C      | 06906fe0    |
| hf_beta   | 🚺       | B              | MM minutes   | C      | 63c0a1a6    |
| hm_omega  | 🚹       | B              | MM minutes   | C      | b55f02a8    |
| hm_psi    | 🚹       | B              | MM minutes   | C      | 2f0f055c    |

---

## 🇮🇹 Italian  
`lang_code='i'` in `misaki["en"]`  
Fallback: `espeak-ng it`

| Name       | Traits   | Target Quality | Duration    | Grade | SHA256     |
|------------|----------|----------------|-------------|--------|-------------|
| if_sara    | 🚺       | B              | MM minutes  | C      | 6c0b253b    |
| im_nicola  | 🚹       | B              | MM minutes  | C      | 234ed066    |

---

## 🇧🇷 Brazilian Portuguese  
`lang_code='p'` in `misaki["en"]`  
Fallback: `espeak-ng pt-br`

| Name      | Traits   | SHA256     |
|-----------|----------|-------------|
| pf_dora   | 🚺       | 07e4ff98    |
| pm_alex   | 🚹       | cf0ba8c5    |
| pm_santa  | 🚹       | d4210316    |

---