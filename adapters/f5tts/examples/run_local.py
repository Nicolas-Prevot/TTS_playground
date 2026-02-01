import sys
from pathlib import Path
from loguru import logger

from tts_adapter_f5tts.adapter import F5TTSAdapter

if __name__ == "__main__":
    # --- 1. Path Setup ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent
    
    # Relative Paths
    SPK_REF_AUDIO = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    SPK_REF_TEXT = "Some call me nature, others call me mother nature."
    
    OUT_DIR = REPO_ROOT / "data" / "local_examples" / "f5tts" 
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Repo Root: {REPO_ROOT}")

    # --- 2. Validation ---
    if not SPK_REF_AUDIO.exists():
        logger.error(f"Speaker reference audio not found at {SPK_REF_AUDIO}")
        sys.exit(1)

    # --- 3. Initialize Adapter ---
    # Using 'vocos' vocoder by default as it is generally faster/supported
    logger.info("Initializing F5-TTS Adapter...")
    tts = F5TTSAdapter(
        model_name="F5TTS_v1_Base",
        vocoder_name="vocos", 
        device="cuda",
    )
    
    tts.load_model()

    # --- 4. Clone Voice ---
    # F5-TTS benefits significantly from having the transcript of the reference audio
    tts.clone_voice(str(SPK_REF_AUDIO), ref_text=SPK_REF_TEXT)

    # --- 5. Feature Showcase ---

    # Case A: Standard Synthesis
    logger.info("--- Case A: Standard Synthesis ---")
    text_base = "I don't really care what you call me. I've been a silent spectator."
    wav_bytes = tts.synthesize(text_base, speed=1.0)
    
    out_path = OUT_DIR / "01_standard.wav"
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved: {out_path.name}")

    # Case B: Speed Control
    logger.info("--- Case B: Fast Speech (Speed 1.2) ---")
    wav_bytes = tts.synthesize(
        "This text should be spoken significantly faster than the previous one.",
        speed=1.2
    )
    
    out_path = OUT_DIR / "02_fast_speech.wav"
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved: {out_path.name}")

    # Case C: Quality vs Speed (NFE Steps)
    logger.info("--- Case C: Low NFE (Fast, lower quality) ---")
    # Lower NFE = fewer diffusion steps = faster generation, potentially metallic artifacts
    wav_bytes = tts.synthesize(
        "Generating this with fewer diffusion steps for speed.",
        nfe_step=16  # Default is usually 32
    )
    
    out_path = OUT_DIR / "03_low_nfe.wav"
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved: {out_path.name}")

    # Case D: Long Text Cross-fading
    logger.info("--- Case D: Long Text with Cross-fade ---")
    long_text = (
        "F5-TTS handles long text by splitting it into chunks. "
        "We can control the cross-fade duration between these chunks to make the transition smoother. "
        "This ensures that longer paragraphs sound cohesive."
    )
    wav_bytes = tts.synthesize(
        long_text,
        cross_fade_duration=0.2  # 200ms overlap
    )
    
    out_path = OUT_DIR / "04_long_crossfade.wav"
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved: {out_path.name}")

    logger.success("All F5-TTS local tests completed.")