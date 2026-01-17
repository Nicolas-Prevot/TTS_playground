import sys
import os
from pathlib import Path
from loguru import logger

# Ensure the adapter src is in path
SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_SRC = SCRIPT_DIR.parent / "src"
sys.path.append(str(ADAPTER_SRC))

from tts_adapter_fish_speech_1_5.adapter import FishSpeech15Adapter

if __name__ == "__main__":
    # --- 1. Path Setup ---
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent
    
    # Relative Paths
    CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "fish-speech-1.5"
    
    # Default filename for the 1.5 decoder
    CODEC_FILENAME = "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    CODEC_PATH = CHECKPOINT_DIR / CODEC_FILENAME
    
    # Hydra config root
    CONFIG_ROOT = REPO_ROOT / "configs" / "fish-speech-1.5"

    SPK_REF_AUDIO = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    SPK_REF_TEXT = "Some call me nature, others call me mother nature."
    
    OUT_DIR = REPO_ROOT / "data" / "local_examples" / "fish_speech_1_5" 
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Repo Root: {REPO_ROOT}")

    # --- 2. Validation ---
    if not CHECKPOINT_DIR.exists():
        logger.error(f"Checkpoint dir not found at {CHECKPOINT_DIR}")
        logger.error("Please run: huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5")
        sys.exit(1)
    
    if not CONFIG_ROOT.exists():
        logger.error(f"Config dir not found at {CONFIG_ROOT}")
        logger.error("Please ensure you created configs/fish-speech-1.5/firefly_gan_vq.yaml")
        sys.exit(1)

    if not SPK_REF_AUDIO.exists():
        logger.error(f"Speaker reference audio not found at {SPK_REF_AUDIO}")
        sys.exit(1)

    # --- 3. Initialize Adapter ---
    logger.info("Initializing Fish Speech 1.5 Adapter...")
    
    tts = FishSpeech15Adapter(
        llama_checkpoint_dir=str(CHECKPOINT_DIR),
        codec_checkpoint_path=str(CODEC_PATH),
        decoder_config_name="firefly_gan_vq",
        config_root_path=str(CONFIG_ROOT),
        device="cuda",
        half=True, 
    )
    
    tts.load_model()

    # --- 4. Clone Voice ---
    logger.info(f"Cloning voice from: {SPK_REF_AUDIO.name}")
    tts.clone_voice(
        ref_audio=str(SPK_REF_AUDIO),
        ref_text=SPK_REF_TEXT
    )

    # --- 5. Feature Showcase ---

    # Case A: Standard Synthesis
    logger.info("--- Case A: Standard Synthesis ---")
    text_a = "Fish Speech 1.5 represents a significant step forward in open source speech generation."
    
    wav_bytes = tts.synthesize(
        text_a,
        max_new_tokens=0,
        top_p=0.7,
        temperature=0.7,
        repetition_penalty=1.2
    )
    
    with open(OUT_DIR / "01_standard.wav", "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved 01_standard.wav")

    # Case B: Long Text (Chunked)
    # 1.5 handles long context better than S1, but chunking is still good for memory
    logger.info("--- Case B: Long Text ---")
    long_text = (
        "The model uses a Firefly GAN based VQ-GAN for high fidelity audio reconstruction. "
        "Coupled with a 7 billion parameter Llama backbone, it understands semantic context "
        "much better than previous iterations, allowing for subtler prosody and intonation changes."
    )
    
    wav_bytes = tts.synthesize(
        long_text,
        chunk_length=200, # iterative prompt length
        temperature=0.7
    )
    
    with open(OUT_DIR / "02_long_text.wav", "wb") as f:
        f.write(wav_bytes)
    logger.success(f"Saved 02_long_text.wav")

    logger.success("All Fish Speech 1.5 local tests completed.")