import sys
import os
from pathlib import Path
from loguru import logger

from tts_adapter_openaudio_s1mini.adapter import OpenAudioS1MiniAdapter

if __name__ == "__main__":
    # --- 1. Path Setup ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent
    
    # Relative Paths
    CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "openaudio-s1-mini"
    CODEC_PATH = CHECKPOINT_DIR / "codec.pth"
    
    # OpenAudio requires a hydra config folder. 
    # We calculate the relative path from this script to the config folder for Hydra.
    # Ideally, this is an absolute path or relative to the running script.
    CONFIG_ROOT = REPO_ROOT / "configs" / "openaudio-s1-mini"

    SPK_REF_AUDIO = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    SPK_REF_TEXT = "Some call me nature, others call me mother nature."
    
    OUT_DIR = REPO_ROOT / "data" / "local_examples" / "openaudio_s1mini" 
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Repo Root: {REPO_ROOT}")

    # --- 2. Validation ---
    if not CHECKPOINT_DIR.exists():
        logger.error(f"Checkpoint dir not found at {CHECKPOINT_DIR}")
        sys.exit(1)
    
    if not CONFIG_ROOT.exists():
        logger.error(f"Config dir not found at {CONFIG_ROOT}")
        sys.exit(1)

    if not SPK_REF_AUDIO.exists():
        logger.error(f"Speaker reference audio not found at {SPK_REF_AUDIO}")
        sys.exit(1)

    # --- 3. Initialize Adapter ---
    logger.info("Initializing OpenAudio S1 Mini Adapter...")
    
    # Hydra expects a path relative to the calling script or an absolute path.
    # We pass the absolute path to the directory containing the yaml.
    tts = OpenAudioS1MiniAdapter(
        llama_checkpoint_dir=str(CHECKPOINT_DIR),
        codec_checkpoint_path=str(CODEC_PATH),
        decoder_config_name="modded_dac_vq",
        config_root_path=str(CONFIG_ROOT), # Explicitly pass config path
        device="cuda",
        half=True, # Use fp16 for speed
    )
    
    tts.load_model()

    # --- 4. Clone Voice ---
    # S1 Mini works best with the transcript of the reference audio
    tts.clone_voice(
        ref_audio=str(SPK_REF_AUDIO),
        ref_text=SPK_REF_TEXT
    )

    # --- 5. Feature Showcase ---

    # Case A: Zero-shot Cloning with Emotion Markers
    logger.info("--- Case A: Emotion Markers ---")
    text_markers = (
        "(shouting)I don't really care what you call me. "
        "(whispering)I've been a silent spectator, watching species evolve. "
        "(laughing) Ha, ha, ha! I am mighty."
    )
    
    wav_bytes = tts.synthesize(
        text_markers,
        max_new_tokens=0, # Let model decide length
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.1
    )
    
    with open(OUT_DIR / "01_emotion_markers.wav", "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved 01_emotion_markers.wav")

    # Case B: Long Text (Chunked Generation)
    # S1 can handle iterative prompting for longer texts
    logger.info("--- Case B: Iterative Prompting (Longer Text) ---")
    long_text = (
        "OpenAudio S1 Mini uses a large language model backbone. "
        "This allows it to understand context better than traditional TTS systems. "
        "It can generate speech in multiple languages including English, Chinese, and Japanese."
    )
    
    wav_bytes = tts.synthesize(
        long_text,
        chunk_length=100, # Generate in semantic chunks
        temperature=0.7
    )
    
    with open(OUT_DIR / "02_iterative.wav", "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved 02_iterative.wav")

    logger.success("All OpenAudio S1 local tests completed.")