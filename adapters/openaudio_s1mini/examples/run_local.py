import sys
from pathlib import Path
from loguru import logger
import torch

from tts_adapter_openaudio_s1mini.adapter import OpenAudioS1MiniAdapter

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent

    CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "openaudio-s1-mini"
    CODEC_PATH = CHECKPOINT_DIR / "codec.pth"
    CONFIG_ROOT = REPO_ROOT / "configs" / "openaudio-s1-mini"

    SPK_REF_AUDIO = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    SPK_REF_TEXT = "Some call me nature, others call me mother nature."

    OUT_DIR = REPO_ROOT / "data" / "local_examples" / "openaudio_s1mini"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = True if device == "cuda" else False

    logger.info(f"Repo Root: {REPO_ROOT}")
    logger.info(f"Device: {device} | half(fp16): {half}")

    if not CHECKPOINT_DIR.exists():
        logger.error(f"Checkpoint dir not found at {CHECKPOINT_DIR}")
        sys.exit(1)
    if not CODEC_PATH.exists():
        logger.error(f"Codec checkpoint not found at {CODEC_PATH}")
        sys.exit(1)
    if not CONFIG_ROOT.exists():
        logger.error(f"Config dir not found at {CONFIG_ROOT}")
        sys.exit(1)
    if not SPK_REF_AUDIO.exists():
        logger.error(f"Speaker reference audio not found at {SPK_REF_AUDIO}")
        sys.exit(1)

    logger.info("Initializing OpenAudio S1 Mini Adapter...")
    tts = OpenAudioS1MiniAdapter(
        llama_checkpoint_dir=str(CHECKPOINT_DIR),
        codec_checkpoint_path=str(CODEC_PATH),
        decoder_config_name="modded_dac_vq",
        config_root_path=str(CONFIG_ROOT),
        device=device,
        half=half,
    )
    tts.load_model()

    tts.clone_voice(ref_audio=str(SPK_REF_AUDIO), ref_text=SPK_REF_TEXT)

    logger.info("--- Case A: Emotion Markers ---")
    text_markers = (
        "(shouting)I don't really care what you call me. "
        "(whispering)I've been a silent spectator, watching species evolve. "
        "(laughing) Ha, ha, ha! I am mighty."
    )
    wav_bytes = tts.synthesize(
        text_markers,
        max_new_tokens=0,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.1,
    )
    (OUT_DIR / "01_emotion_markers.wav").write_bytes(wav_bytes)
    logger.success("Saved 01_emotion_markers.wav")

    logger.info("--- Case B: Iterative Prompting (Longer Text) ---")
    long_text = (
        "OpenAudio S1 Mini uses a large language model backbone. "
        "This allows it to understand context better than traditional TTS systems. "
        "It can generate speech in multiple languages including English, Chinese, and Japanese."
    )
    wav_bytes = tts.synthesize(long_text, chunk_length=100, temperature=0.7)
    (OUT_DIR / "02_iterative.wav").write_bytes(wav_bytes)
    logger.success("Saved 02_iterative.wav")

    logger.success("All OpenAudio S1 local tests completed.")
