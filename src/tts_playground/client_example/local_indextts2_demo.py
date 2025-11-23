from pathlib import Path
import os
import base64  # only for a quick size print at the end (optional)

from tts_playground.adapters.IndexTTS2 import IndexTTS2Adapter

if __name__ == "__main__":
    # ---------- paths ----------
    OUT_DIR = Path("data/gen_local")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    SPK_REF_PATH = "data/ref/basic_ref_en.wav"
    EMO_REF_PATH = "data/ref/emo_hate.wav"

    assert os.path.isfile(SPK_REF_PATH), f"Missing speaker ref: {SPK_REF_PATH}"
    assert os.path.isfile(EMO_REF_PATH), f"Missing emotion ref: {EMO_REF_PATH}"

    # ---------- texts (same as client demo) ----------
    short_txt = "Hello! This is IndexTTS2 via TTS_PLAYGROUND."
    long_txt = (
        "This is a slightly longer passage designed to cross a few sentence boundaries. "
        "It helps us hear how the model handles segmentation, pacing, and the inter-segment silence. "
        "We will also vary sampling parameters to hear more expressive delivery."
    )

    # ---------- adapter init (match client/base_init) ----------
    tts = IndexTTS2Adapter(
        model_dir="checkpoints/indextts2",
        cfg_path="checkpoints/indextts2/config.yaml",
        use_fp16=False,
        device=None,
        # generation defaults (can be overridden per-case)
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        max_text_tokens_per_segment=120,
        verbose=True,  # enable so you can see emo_* reflected in logs
    )
    tts.load_model()
    tts.clone_voice(SPK_REF_PATH)

    def run_case(name: str, text: str, kwargs: dict):
        print(f"\n=== Local case: {name} ===")
        # IMPORTANT: do not mutate adapter emotion state across cases; pass via kwargs.
        wav_bytes = tts.synthesize(text, **kwargs)
        out = OUT_DIR / f"indextts2_{name}.wav"
        out.write_bytes(wav_bytes)
        print(f"[{name}] OK → {out} (bytes={len(wav_bytes)})")

    # 1) Baseline: emotion from speaker voice (no emo_* passed), deterministic
    run_case(
        "baseline_speaker_emotion",
        short_txt,
        {
            "use_random": False,
        },
    )

    # 2) Emotion from separate audio, moderate alpha (audible, subtle)
    run_case(
        "emo_audio_prompt_alpha07",
        short_txt,
        {
            "emo_audio_prompt": EMO_REF_PATH,
            "emo_alpha": 0.7,
            "use_random": False,
        },
    )

    # 3) Emotion from separate audio, strong alpha (max effect)
    run_case(
        "emo_audio_prompt_alpha10",
        short_txt,
        {
            "emo_audio_prompt": EMO_REF_PATH,
            "emo_alpha": 1.0,
            "use_random": False,
        },
    )

    # 4) Emotion from vector (happy + calm), deterministic
    run_case(
        "emo_vector_happy_calm",
        short_txt,
        {
            "emo_vector": [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6],
            "emo_alpha": 1.0,
            "use_random": False,
            # optional sampling tweaks (kept same as client demo idea)
            # "temperature": 0.7,
            # "top_p": 0.92,
        },
    )

    # 5) Emotion from natural language text (QwenEmotion), moderate alpha
    run_case(
        "emo_text_melancholic",
        "A short melancholic passage, softly spoken.",
        {
            "use_emo_text": True,
            "emo_text": "gentle, slightly melancholic, intimate",
            "emo_alpha": 0.8,
            "use_random": False,
        },
    )

    # 6) Longer text with forced segmentation and longer pauses
    run_case(
        "segmentation_interval300_ms",
        long_txt,
        {
            "max_text_tokens_per_segment": 40,  # force smaller segments
            "interval_silence": 300,            # 300 ms pause between segments
            "use_random": False,
        },
    )

    # 7) More expressive sampling (to hear style variability)
    run_case(
        "expressive_sampling",
        long_txt,
        {
            "do_sample": True,
            "temperature": 0.95,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 8.0,
            "use_random": True,
        },
    )

    print("\nAll local cases finished. Compare with API outputs in data/gen_api/")
