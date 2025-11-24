import sys
from pathlib import Path
from loguru import logger

from tts_adapter_indextts2.adapter import IndexTTS2Adapter


if __name__ == "__main__":
    # --- 1. Robust Path Setup ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent
    
    # Paths relative to Repo Root
    CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "indextts2"
    CONFIG_PATH = CHECKPOINT_DIR / "config.yaml"
    
    # Reference Audio Files
    SPK_REF_AUDIO = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    EMO_REF_AUDIO = REPO_ROOT / "data" / "ref" / "emo_hate.wav"
    
    # Output Directory
    OUT_DIR = REPO_ROOT / "data" / "local_examples" / "indextts2" 
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Repo Root: {REPO_ROOT}")
    logger.info(f"Checkpoints: {CHECKPOINT_DIR}")

    # --- 2. Validation ---
    if not CHECKPOINT_DIR.exists():
        logger.error(f"Checkpoint dir not found at {CHECKPOINT_DIR}")
        logger.info("Please ensure you have downloaded the models into the checkpoints folder.")
        sys.exit(1)
    
    if not CONFIG_PATH.exists():
        logger.error(f"Config not found at {CONFIG_PATH}")
        sys.exit(1)

    if not SPK_REF_AUDIO.exists():
        logger.error(f"Speaker reference audio not found at {SPK_REF_AUDIO}")
        sys.exit(1)

    # --- 3. Initialize Adapter ---
    logger.info("Initializing IndexTTS2 Adapter...")
    
    # We can capture the adapter's internal print statements if we wanted, 
    # but here we just log around it.
    tts = IndexTTS2Adapter(
        model_dir=str(CHECKPOINT_DIR),
        cfg_path=str(CONFIG_PATH),
        use_fp16=False,
        device=None, 
        verbose=False, # Set False to keep console clean, let loguru handle our logs
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
    )
    
    tts.load_model()
    logger.success("Model loaded successfully.")

    # --- 4. Clone Voice ---
    tts.clone_voice(str(SPK_REF_AUDIO))
    logger.info(f"Voice cloned from {SPK_REF_AUDIO.name}")

    # --- 5. Feature Showcase ---

    # Case A: Baseline
    logger.info("--- Case A: Baseline (Natural Speaker Emotion) ---")
    text_base = "Hello! This is a baseline test of IndexTTS2 zero-shot voice cloning."
    wav_bytes = tts.synthesize(text_base)
    
    out_path = OUT_DIR / "01_baseline.wav"
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved: {out_path.name}")

    # Case B: Emotion Reference
    if EMO_REF_AUDIO.exists():
        logger.info(f"--- Case B: Emotion Transfer ({EMO_REF_AUDIO.name}) ---")
        text_emo = "I cannot believe you did that! This is absolutely unacceptable!"
        
        wav_bytes = tts.synthesize(
            text_emo,
            emo_audio_prompt=str(EMO_REF_AUDIO),
            emo_alpha=0.8
        )
        
        out_path = OUT_DIR / "02_emotion_ref_hate.wav"
        with open(out_path, "wb") as f:
            f.write(wav_bytes)
        logger.success(f"Saved: {out_path.name}")
    else:
        logger.warning(f"Skipping Case B: {EMO_REF_AUDIO.name} not found.")

    # Case C: Vector Control
    logger.info("--- Case C: Emotion Vector (Happy/Calm mix) ---")
    text_vec = "I am feeling quite relaxed and happy today. The sun is shining."
    
    wav_bytes = tts.synthesize(
        text_vec,
        emo_vector=[0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6], # Happy/Calm
        emo_alpha=1.0,
        use_random=False,
        temperature=0.7
    )
    
    out_path = OUT_DIR / "03_emotion_vector_happy.wav"
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved: {out_path.name}")

    # Case D: Text Description
    logger.info("--- Case D: Natural Language Emotion Description ---")
    text_desc = "It's a dark and stormy night... I feel so alone."
    
    wav_bytes = tts.synthesize(
        text_desc,
        use_emo_text=True,
        emo_text="Sad, melancholic, slightly afraid, whispering",
        emo_alpha=0.9
    )
    
    out_path = OUT_DIR / "04_emotion_description_sad.wav"
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved: {out_path.name}")

    # Case E: Advanced Segmentation
    logger.info("--- Case E: Long Text & Sampling Tweaks ---")
    long_text = (
        "This is a longer passage designed to test sentence segmentation. "
        "The model should handle pauses naturally between these sentences. "
        "We are also using a higher temperature for more expressive variability."
    )
    
    wav_bytes = tts.synthesize(
        long_text,
        temperature=0.95,
        top_p=0.95,
        repetition_penalty=12.0,
        max_text_tokens_per_segment=100,
        interval_silence=400
    )
    
    out_path = OUT_DIR / "05_long_text_expressive.wav"
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved: {out_path.name}")

    logger.success("All tests completed successfully.")