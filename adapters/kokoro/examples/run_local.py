import sys
from pathlib import Path
from loguru import logger

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
    logger.info("Initializing Kokoro Adapter...")
    tts = KokoroTTSAdapter(lang_code="a", voice="af_heart")
    tts.load_model()

    # --- 3. Feature Showcase ---

    # Case A: American English (af_heart)
    logger.info("--- Case A: American English (af_heart) ---")
    tts.clone_voice(voice="af_heart", lang_code="a")

    wav_bytes = tts.synthesize(
        "I don't really care what you call me. I've been a silent spectator.",
        speed=1.0,
        split_pattern=r"\n+",
    )

    (OUT_DIR / "01_us_heart.wav").write_bytes(wav_bytes)
    logger.success("Saved 01_us_heart.wav")

    # Case B: British English (bm_george)
    # Upstream recommends matching lang_code with voice.
    logger.info("--- Case B: British English (bm_george) - Slow ---")
    tts.clone_voice(voice="bm_george", lang_code="b")

    wav_bytes = tts.synthesize(
        "Empires rise and fall. But always remember, I am mighty and enduring.",
        speed=0.8,
        split_pattern=r"\n+",
    )

    (OUT_DIR / "02_uk_george_slow.wav").write_bytes(wav_bytes)
    logger.success("Saved 02_uk_george_slow.wav")

    # Case C: French (ff_siwis)
    # Note: non-English languages often require espeak-ng installed.
    logger.info("--- Case C: French (ff_siwis) ---")
    tts.clone_voice(voice="ff_siwis", lang_code="f")

    wav_bytes = tts.synthesize(
        "À l'époque classique, à Athènes, les auteurs doivent présenter au concours trois tragédies.",
        speed=1.0,
        split_pattern=r"\n+",
    )

    (OUT_DIR / "03_french_siwis.wav").write_bytes(wav_bytes)
    logger.success("Saved 03_french_siwis.wav")

    logger.success("Kokoro local tests completed.")
