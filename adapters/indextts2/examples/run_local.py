import sys
from pathlib import Path
from loguru import logger

from tts_adapter_indextts2.adapter import IndexTTS2Adapter

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent

    CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "indextts2"
    CONFIG_PATH = CHECKPOINT_DIR / "config.yaml"

    SPK_REF_AUDIO = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    EMO_REF_AUDIO = REPO_ROOT / "data" / "ref" / "emo_hate.wav"

    OUT_DIR = REPO_ROOT / "data" / "local_examples" / "indextts2"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = None          # e.g. "cuda:0" or "cpu" or None for auto
    USE_FP16 = False

    logger.info(f"Repo Root: {REPO_ROOT}")
    logger.info(f"Checkpoints: {CHECKPOINT_DIR}")

    if not CHECKPOINT_DIR.exists():
        logger.error(f"Checkpoint dir not found at {CHECKPOINT_DIR}")
        sys.exit(1)
    if not CONFIG_PATH.exists():
        logger.error(f"Config not found at {CONFIG_PATH}")
        sys.exit(1)
    if not SPK_REF_AUDIO.exists():
        logger.error(f"Speaker reference audio not found at {SPK_REF_AUDIO}")
        sys.exit(1)

    logger.info("Initializing IndexTTS2 Adapter...")
    tts = IndexTTS2Adapter(
        model_dir=str(CHECKPOINT_DIR),
        cfg_path=str(CONFIG_PATH),
        use_fp16=USE_FP16,
        device=DEVICE,
        verbose=False,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
    )

    tts.load_model()
    logger.success("Model loaded successfully.")

    tts.clone_voice(str(SPK_REF_AUDIO))
    logger.info(f"Voice cloned from {SPK_REF_AUDIO.name}")

    out_path = OUT_DIR / "01_baseline.wav"
    out_path.write_bytes(tts.synthesize("Hello! This is a baseline test of IndexTTS2."))
    logger.success(f"Saved: {out_path.name}")

    if EMO_REF_AUDIO.exists():
        out_path = OUT_DIR / "02_emotion_ref.wav"
        out_path.write_bytes(
            tts.synthesize(
                "I cannot believe you did that! This is absolutely unacceptable!",
                emo_audio_prompt=str(EMO_REF_AUDIO),
                emo_alpha=0.8,
            )
        )
        logger.success(f"Saved: {out_path.name}")
    else:
        logger.warning(f"Skipping emotion test: {EMO_REF_AUDIO.name} not found.")

    logger.success("Done.")
