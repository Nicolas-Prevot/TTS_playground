import sys
import base64
from pathlib import Path
from typing import Any, Dict

# Add the source root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "src"))

from tts_playground.client.tts_client import TTSClient

if __name__ == "__main__":
    BASE_URL = "http://localhost:7000"
    
    OUT_DIR = REPO_ROOT / "data" / "api_examples" / "openaudio_s1mini"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference Audio
    SPK_REF_PATH = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    SPK_REF_TEXT = "Some call me nature, others call me mother nature."

    if not SPK_REF_PATH.exists():
        print(f"ERROR: Reference file not found: {SPK_REF_PATH}")
        sys.exit(1)

    # Initialize Client
    print(f"Connecting to {BASE_URL}...")
    client = TTSClient(BASE_URL, timeout=300.0)

    # --- 1. Prepare Blob ---
    ref_blob = client.pack_file(str(SPK_REF_PATH))

    # --- 2. Configuration ---
    # Server-side paths relative to where the worker runs
    base_init = {
        "llama_checkpoint_dir": "checkpoints/openaudio-s1-mini",
        "codec_checkpoint_path": "checkpoints/openaudio-s1-mini/codec.pth",
        "decoder_config_name": "modded_dac_vq",
        # Ideally absolute path inside container, or relative to adapter root
        # The adapter defaults usually work if the folder structure is mounted correctly
        "config_root_path": "/workspace/configs/openaudio-s1-mini", 
        "device": "cuda",
        "half": True
    }

    def run_case(name: str, text: str, kwargs: Dict[str, Any]):
        print(f"\n=== Running API Case: {name} ===")
        dest = OUT_DIR / f"{name}.wav"
        
        try:
            res = client.synth(
                adapter="openaudios1mini",
                init=base_init,
                load_model={},
                clone_voice={
                    "ref_audio": ref_blob,
                    "ref_text": SPK_REF_TEXT
                },
                synthesize={"text": text, "kwargs": kwargs},
                dest_path=str(dest),
                wait=True,
                download=True
            )
            
            if res.get("state") == "SUCCESS":
                file_size = dest.stat().st_size if dest.exists() else 0
                print(f"[{name}] SUCCESS -> {dest.name} ({file_size} bytes)")
            else:
                print(f"[{name}] FAILED: {res.get('error')}")
                
        except Exception as e:
            print(f"[{name}] ERROR: {str(e)}")

    try:
        # --- Case 1: Emotional Speech ---
        run_case(
            "01_emotional",
            "(shouting) Hello there! (laughing) I am generating this via the API!",
            {
                "temperature": 0.8,
                "repetition_penalty": 1.1,
                "max_new_tokens": 0
            }
        )

        # --- Case 2: Creative Writing ---
        run_case(
            "02_creative",
            "(whispering) The secret to high quality TTS is... (normal) precise control over the latent space.",
            {
                "top_p": 0.9,
                "temperature": 0.7
            }
        )

    finally:
        client.close()
        print("\nDone.")