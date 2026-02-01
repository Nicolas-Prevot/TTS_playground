from pathlib import Path
from loguru import logger

from tts_adapter_dia2.adapter import Dia2Adapter


if __name__ == "__main__":
    # --- 1. Paths ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent

    OUT_DIR = REPO_ROOT / "data" / "local_examples" / "dia2"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Re-use the same reference audio as other adapters if available
    SPK_REF_AUDIO = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"

    logger.info(f"Repo Root: {REPO_ROOT}")
    if not SPK_REF_AUDIO.exists():
        logger.warning(
            f"Reference audio not found at {SPK_REF_AUDIO}. "
            "Continuing without audio prompts."
        )

    # --- 2. Initialize Adapter ---
    logger.info("Initializing Dia2 Adapter...")
    tts = Dia2Adapter(
        repo_id="nari-labs/Dia2-2B",
        device=None,         # auto-select (cuda if available, else cpu)
        dtype="bfloat16",    # recommended for CUDA GPUs; auto-fallback to float32 on CPU
        cfg_scale=6.0,       # matches upstream CLI quickstart
        audio_temperature=0.8,
        audio_top_k=50,
        use_cuda_graph=True, # only used when device is CUDA
    )
    tts.load_model()

    # --- 3. Optional Voice Conditioning (Audio Prompt) ---
    if SPK_REF_AUDIO.exists():
        logger.info(
            "Setting prefix speaker 1 from reference audio (conditioning). "
            "NOTE: Dia2 uses Whisper to transcribe prefix audio (extra latency)."
        )
        tts.clone_voice(
            prefix_speaker_1=str(SPK_REF_AUDIO),
            include_prefix=False,  # conditioning only (do not prepend ref audio)
        )

    # --- 4. Example A: Basic Two-Speaker Dialogue ---
    logger.info("--- Case A: Basic dialogue ---")
    script_a = (
        "[S1] Hello, this is Dia2 running inside TTS Playground.\n"
        "[S2] Nice! We can generate natural dialogue directly from a script."
    )

    wav_bytes = tts.synthesize(script_a)
    out_path = OUT_DIR / "01_basic_dialogue.wav"
    out_path.write_bytes(wav_bytes)
    logger.success(f"Saved: {out_path}")

    # --- 5. Example B: Higher temperature / more variation ---
    logger.info("--- Case B: Higher temperature ---")
    script_b = (
        "[S1] Let's try a slightly more expressive setting.\n"
        "[S2] Sure, increasing the temperature should give more variation."
    )

    wav_bytes = tts.synthesize(
        script_b,
        temperature=0.95,
        top_k=80,
    )
    out_path = OUT_DIR / "02_high_temp.wav"
    out_path.write_bytes(wav_bytes)
    logger.success(f"Saved: {out_path}")

    # --- 6. Example C: Using only text (no audio prompt) ---
    logger.info("--- Case C: No prefix audio ---")
    # Disable cached prefix prompts for one call by passing None explicitly.
    script_c = (
        "[S1] This line is generated without using any audio prefix.\n"
        "[S2] Dia2 still produces natural dialogue from text alone."
    )

    wav_bytes = tts.synthesize(
        script_c,
        prefix_speaker_1=None,
        prefix_speaker_2=None,
    )
    out_path = OUT_DIR / "03_no_prefix.wav"
    out_path.write_bytes(wav_bytes)
    logger.success(f"Saved: {out_path}")

    logger.success("All Dia2 local tests completed.")
