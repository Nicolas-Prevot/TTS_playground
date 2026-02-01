from pathlib import Path
from loguru import logger

from tts_adapter_kyutai.adapter import KyutaiTTSAdapter


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent

    OUT_DIR = REPO_ROOT / "data" / "local_examples" / "kyutai"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Repo Root: {REPO_ROOT}")

    logger.info("Initializing Kyutai Adapter...")
    tts = KyutaiTTSAdapter(device="cuda", temp=0.6, cfg_coef=1.0)
    tts.load_model()
    logger.success("Model loaded.")

    # Case A: English (VCTK)
    logger.info("--- Case A: English Synthesis (VCTK Voice) ---")
    tts.clone_voice("vctk/p225_023.wav")

    text_en = (
        "I don't really care what you call me. I've been a silent spectator, "
        "watching species evolve, empires rise and fall."
    )
    wav_bytes = tts.synthesize(text_en)
    out_path = OUT_DIR / "01_english_vctk.wav"
    out_path.write_bytes(wav_bytes)
    logger.success(f"Saved: {out_path.name}")

    # Case B: French (CML-TTS)
    logger.info("--- Case B: French Synthesis (CML Voice) ---")
    tts.clone_voice("cml-tts/fr/4724_3731_000031-0001.wav")

    text_fr = (
        "À l'époque classique, à Athènes, les auteurs doivent présenter "
        "au concours trois tragédies."
    )
    wav_bytes = tts.synthesize(text_fr)
    out_path = OUT_DIR / "02_french_cml.wav"
    out_path.write_bytes(wav_bytes)
    logger.success(f"Saved: {out_path.name}")

    # Case C: Per-call conditioning strength (cfg_coef)
    logger.info("--- Case C: Per-call cfg_coef (conditioning strength) ---")
    tts.clone_voice("expresso/ex01-ex02_default_001_channel1_168s.wav")

    text_var = "Hey, did you see that? That was absolutely incredible!"
    wav_bytes = tts.synthesize(text_var, cfg_coef=2.0)
    out_path = OUT_DIR / "03_english_expresso_cfg2.wav"
    out_path.write_bytes(wav_bytes)
    logger.success(f"Saved: {out_path.name}")

    logger.success("Kyutai local tests completed.")
