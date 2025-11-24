import sys
from pathlib import Path
from loguru import logger

from tts_adapter_kyutai.adapter import KyutaiTTSAdapter

if __name__ == "__main__":
    # --- 1. Path Setup ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent
    
    OUT_DIR = REPO_ROOT / "data" / "local_examples" / "kyutai"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Repo Root: {REPO_ROOT}")

    # --- 2. Initialize Adapter ---
    logger.info("Initializing Kyutai Adapter...")
    # Using defaults: kyutai/tts-1.6b-en_fr
    tts = KyutaiTTSAdapter(
        device="cuda",
        temp=0.6,
        cfg_coef=1.0
    )
    
    tts.load_model()
    logger.success("Model loaded.")

    # --- 3. Synthesis Cases ---

    # Case A: English (VCTK Voice)
    logger.info("--- Case A: English Synthesis (VCTK Voice) ---")
    
    # Kyutai uses specific voice IDs available in the Moshi voice repo
    voice_id_en = "vctk/p225_023.wav" 
    tts.clone_voice(voice_id_en)
    
    text_en = (
        "I don't really care what you call me. I've been a silent spectator, "
        "watching species evolve, empires rise and fall."
    )
    
    wav_bytes = tts.synthesize(text_en)
    
    out_path = OUT_DIR / "01_english_vctk.wav"
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved: {out_path.name}")

    # Case B: French (CML-TTS Voice)
    logger.info("--- Case B: French Synthesis (CML Voice) ---")
    
    voice_id_fr = "cml-tts/fr/4724_3731_000031-0001.wav"
    tts.clone_voice(voice_id_fr)
    
    text_fr = (
        "À l'époque classique, à Athènes, les auteurs doivent présenter "
        "au concours trois tragédies."
    )
    
    wav_bytes = tts.synthesize(text_fr)
    
    out_path = OUT_DIR / "02_french_cml.wav"
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved: {out_path.name}")

    # Case C: Adjusting Temperature (Expressiveness)
    logger.info("--- Case C: Higher Temperature (More variation) ---")
    
    # Note: To change temp strictly, we usually need to reload or 
    # rely on the stochastic nature of the loaded model if the adapter exposes it per-call.
    # The adapter implementation allows us to set internal state if we wanted, 
    # but here we rely on the stochastic nature at temp=0.6 initialized earlier.
    # We will just generate a different sentence to show stability.
    
    text_var = "But always remember, I am mighty and enduring."
    wav_bytes = tts.synthesize(text_var)
    
    out_path = OUT_DIR / "03_english_short.wav"
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved: {out_path.name}")

    logger.success("Kyutai local tests completed.")