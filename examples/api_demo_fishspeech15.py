import sys
from pathlib import Path
from typing import Any, Dict

# Add the source root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "src"))

from tts_playground.client.tts_client import TTSClient


if __name__ == "__main__":
    BASE_URL = "http://localhost:7000"

    OUT_DIR = REPO_ROOT / "data" / "api_examples" / "fishspeech15"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference audio
    SPK_REF_PATH = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    SPK_REF_TEXT = "Some call me nature, others call me mother nature."

    if not SPK_REF_PATH.exists():
        print(f"ERROR: Reference file not found: {SPK_REF_PATH}")
        sys.exit(1)

    print(f"Connecting to {BASE_URL}...")
    client = TTSClient(BASE_URL, timeout=180.0)

    try:
        ref_blob = client.pack_file(str(SPK_REF_PATH))

        base_init = {
            "llama_checkpoint_dir": "/workspace/checkpoints/fish-speech-1.5",
            "codec_checkpoint_path": "/workspace/checkpoints/fish-speech-1.5/"
                                      "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
            "decoder_config_name": "firefly_gan_vq",
            "config_root_path": "/workspace/configs/fish-speech-1.5",
            "device": "cuda",
            "half": True,
        }

        def run_case(name: str, text: str, kwargs: Dict[str, Any]):
            print(f"\n=== Running API Case: {name} ===")
            dest = OUT_DIR / f"{name}.wav"

            try:
                res = client.synth(
                    adapter="fishspeech15",
                    init=base_init,
                    load_model={},
                    clone_voice={
                        "ref_audio": ref_blob,
                        "ref_text": SPK_REF_TEXT,
                    },
                    synthesize={"text": text, "kwargs": kwargs},
                    dest_path=str(dest),
                    wait=True,
                    download=True,
                )

                if res.get("state") == "SUCCESS":
                    file_size = dest.stat().st_size if dest.exists() else 0
                    print(f"[{name}] SUCCESS -> {dest.name} ({file_size} bytes)")
                else:
                    print(f"[{name}] FAILED: {res.get('error')}")
            except Exception as e:
                print(f"[{name}] ERROR: {str(e)}")

        # Case 1: Neutral
        run_case(
            "01_neutral",
            "Hello! This is Fish Speech 1.5 integrated through the Playground API.",
            {"top_p": 0.8, "temperature": 0.7},
        )

        # Case 2: Longer text / chunked
        long_text = (
            "This is a longer test sentence for Fish Speech 1.5. "
            "The TTS Playground lets us compare this model against the others."
        )
        run_case(
            "02_long",
            long_text,
            {"chunk_length": 150, "temperature": 0.75},
        )

    finally:
        client.close()
        print("\nDone.")
