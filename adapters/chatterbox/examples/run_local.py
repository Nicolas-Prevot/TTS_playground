import sys
from pathlib import Path
from loguru import logger

from tts_adapter_chatterbox.adapter import ChatterboxTTSAdapter

if __name__ == "__main__":
    # --- 1. Path Setup ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent
    
    SPK_REF_AUDIO = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    
    OUT_DIR = REPO_ROOT / "data" / "local_examples" / "chatterbox"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Repo Root: {REPO_ROOT}")

    # --- 2. Validation ---
    if not SPK_REF_AUDIO.exists():
        logger.error(f"Speaker reference audio not found at {SPK_REF_AUDIO}")
        sys.exit(1)

    # --- 3. Initialize & Load ---
    tts = ChatterboxTTSAdapter()
    tts.load_model()

    # --- 4. Clone Voice ---
    tts.clone_voice(str(SPK_REF_AUDIO))

    # --- 5. Synthesize ---
    
    # Case A: Neutral / Default
    logger.info("Generating Case A: Neutral...")
    audio_bytes = tts.synthesize(
        "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall.",
        cfg_weight=0.5,
        exaggeration=0.5, # Neutral
    )
    with open(OUT_DIR / "01_neutral.wav", "wb") as f:
        f.write(audio_bytes)

    # Case B: High Emotion (Exaggerated)
    logger.info("Generating Case B: High Emotion...")
    audio_bytes = tts.synthesize(
        "But always remember, I am mighty and enduring!",
        cfg_weight=0.6,
        exaggeration=0.8, # More dramatic
        temperature=0.9
    )
    with open(OUT_DIR / "02_emotional.wav", "wb") as f:
        f.write(audio_bytes)

    logger.success("Chatterbox local tests completed.")