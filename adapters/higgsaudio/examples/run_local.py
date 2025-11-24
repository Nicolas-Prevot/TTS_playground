import sys
from pathlib import Path
from loguru import logger

from tts_adapter_higgsaudio.adapter import HiggsAudioAdapter

if __name__ == "__main__":
    # --- 1. Path Setup ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent
    
    SPK_REF_AUDIO = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    SPK_REF_TEXT = "Some call me nature, others call me mother nature."
    SCENE_DESC = "A clear voice speaking in a quiet room."
    
    OUT_DIR = REPO_ROOT / "data" / "local_examples" / "higgsaudio"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Repo Root: {REPO_ROOT}")

    # --- 2. Validation ---
    if not SPK_REF_AUDIO.exists():
        logger.error(f"Speaker reference audio not found at {SPK_REF_AUDIO}")
        sys.exit(1)

    # --- 3. Initialize Adapter ---
    # Ensure you have 'bosonai/higgs-audio-v2-generation-3B-base' and tokenizer downloaded via HF
    tts = HiggsAudioAdapter(
        device="cuda" if sys.platform != "darwin" else "mps",
        use_static_kv_cache=True,
        max_new_tokens=4096
    )
    
    tts.load_model()
    logger.success("Model loaded.")

    # --- 4. Clone Voice ---
    tts.clone_voice(
        ref_audio=str(SPK_REF_AUDIO),
        ref_text=SPK_REF_TEXT,
        scene_prompt=SCENE_DESC
    )
    logger.info("Voice context set.")

    # --- 5. Synthesize ---
    logger.info("Generating Audio...")
    
    text = (
        "Hello! This is HiggsAudio running locally via the adapter. "
        "It supports high-fidelity voice cloning and expressive speech."
    )
    
    audio_bytes = tts.synthesize(
        text,
        temperature=0.95,
        top_p=0.9,
        seed=42
    )

    out_path = OUT_DIR / "higgs_output.wav"
    with open(out_path, "wb") as f:
        f.write(audio_bytes)
        
    logger.success(f"Saved to {out_path}")