import sys
import os
from pathlib import Path
from loguru import logger

from tts_adapter_vibevoice.adapter import VibeVoiceAdapter

if __name__ == "__main__":
    # --- 1. Path Setup ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent
    
    # Relative Paths
    SPK_REF_AUDIO = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    FR_REF_AUDIO = REPO_ROOT / "data" / "ref" / "fr" / "Ellie_Bishop_fr.wav"
    
    OUT_DIR = REPO_ROOT / "data" / "local_examples" / "vibevoice" 
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Repo Root: {REPO_ROOT}")

    # --- 2. Validation ---
    if not SPK_REF_AUDIO.exists():
        logger.error(f"Speaker reference audio not found at {SPK_REF_AUDIO}")
        sys.exit(1)

    # --- 3. Initialize Adapter ---
    # Use 1.5B for lighter testing, or 7B for quality
    logger.info("Initializing VibeVoice Adapter...")
    tts = VibeVoiceAdapter(
        model_id="vibevoice/VibeVoice-7B", 
        device="cuda",
        verbose=True
    )
    tts.load_model()

    # --- Case A: Single Speaker Clone ---
    logger.info("--- Case A: Single Speaker Clone ---")
    tts.clone_voice(str(SPK_REF_AUDIO))
    
    wav_bytes = tts.synthesize(
        "Hello! I am cloning this voice using VibeVoice. The quality should be quite high.",
        cfg_scale=1.3
    )
    
    out_path = OUT_DIR / "01_single_speaker.wav"
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved: {out_path.name}")

    # --- Case B: Non-Cloned (No Prefill) ---
    # Uses the model's internal knowledge without reference conditioning
    logger.info("--- Case B: Non-Cloned (Prefill OFF) ---")
    wav_bytes = tts.synthesize(
        "This is generated without looking at the reference audio.",
        is_prefill=False,
        generation_config={"do_sample": True, "temperature": 0.8}
    )
    
    out_path = OUT_DIR / "02_no_prefill.wav"
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved: {out_path.name}")

    # --- Case C: Multi-Speaker Dialogue ---
    if FR_REF_AUDIO.exists():
        logger.info("--- Case C: Multi-Speaker Dialogue ---")
        script = (
            "Speaker 1: Hello, how are you today?\n"
            "Speaker 2: Bonjour! Je vais très bien, merci.\n"
            "Speaker 1: That is great to hear!"
        )
        
        # Explicit mapping: "1" -> English ref, "2" -> French ref
        speaker_map = {
            "1": str(SPK_REF_AUDIO),
            "2": str(FR_REF_AUDIO)
        }
        
        # Update clone config for this call (using keyword override in synthesize if preferred, 
        # but here we reset clone state for clarity)
        tts.clone_voice(speaker_voices=speaker_map)
        
        wav_bytes = tts.synthesize(script, cfg_scale=1.2)
        
        out_path = OUT_DIR / "03_dialogue.wav"
        with open(out_path, "wb") as f:
            f.write(wav_bytes)
        logger.success(f"Saved: {out_path.name}")
    else:
        logger.warning("Skipping multi-speaker test (French reference not found).")

    logger.success("All VibeVoice local tests completed.")