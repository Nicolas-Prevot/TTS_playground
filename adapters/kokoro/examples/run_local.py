import sys
from pathlib import Path
from loguru import logger

# Add the adapter source to sys.path to run directly
SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_SRC = SCRIPT_DIR.parent / "src"
sys.path.append(str(ADAPTER_SRC))

from tts_adapter_kokoro.adapter import KokoroTTSAdapter

if __name__ == "__main__":
    # --- 1. Path Setup ---
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent
    OUT_DIR = REPO_ROOT / "data" / "local_examples" / "kokoro"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Repo Root: {REPO_ROOT}")

    # --- 2. Initialize Adapter ---
    # Kokoro is very lightweight (82M params).
    # We start with American English ('a').
    logger.info("Initializing Kokoro Adapter...")
    tts = KokoroTTSAdapter(lang_code="a", voice="af_heart")
    tts.load_model()

    # --- 3. Feature Showcase ---

    # Case A: Standard American Female (af_heart)
    logger.info("--- Case A: Standard American (af_heart) ---")
    text_us = "I don't really care what you call me. I've been a silent spectator."
    
    # Note: clone_voice here just sets the internal voice ID string
    tts.clone_voice(voice="af_heart", lang_code="a")
    
    wav_bytes = tts.synthesize(text_us, speed=1.0)
    
    with open(OUT_DIR / "01_us_heart.wav", "wb") as f:
        f.write(wav_bytes)
    logger.success("Saved 01_us_heart.wav")

    # Case B: British Male (bm_george) + Slower Speed
    logger.info("--- Case B: British Male (bm_george) - Slow ---")
    
    # Switching voice ID. Lang code 'b' is often used for British in Kokoro/Misaki contexts,
    # though 'a' (English) pipeline often handles both. We stick to 'a' for pipeline stability 
    # unless a strict switch is needed.
    tts.clone_voice(voice="bm_george", lang_code="a") 
    
    wav_bytes = tts.synthesize(
        "Empires rise and fall. But always remember, I am mighty and enduring.",
        speed=0.8
    )
    
    with open(OUT_DIR / "02_uk_george_slow.wav", "wb") as f:
        f.write(wav_bytes)
    logger.success("Saved 02_uk_george_slow.wav")

    # Case C: French (ff_siwis) - Switching Language
    # This requires the adapter to reload the pipeline internally for 'f'
    logger.info("--- Case C: French (ff_siwis) ---")
    
    tts.clone_voice(voice="ff_siwis", lang_code="f")
    
    text_fr = (
        "À l'époque classique, à Athènes, les auteurs doivent présenter "
        "au concours trois tragédies."
    )
    
    wav_bytes = tts.synthesize(text_fr, speed=1.0)
    
    with open(OUT_DIR / "03_french_siwis.wav", "wb") as f:
        f.write(wav_bytes)
    logger.success("Saved 03_french_siwis.wav")

    logger.success("Kokoro local tests completed.")