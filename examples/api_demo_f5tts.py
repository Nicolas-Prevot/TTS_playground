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
    
    # Output Directory
    OUT_DIR = REPO_ROOT / "data" / "api_examples" / "f5tts"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference audio
    SPK_REF_PATH = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    SPK_REF_TEXT = "Some call me nature, others call me mother nature."

    if not SPK_REF_PATH.exists():
        print(f"ERROR: Reference file not found: {SPK_REF_PATH}")
        sys.exit(1)

    # Initialize Client
    print(f"Connecting to {BASE_URL}...")
    client = TTSClient(BASE_URL, timeout=180.0) # F5 might be slower on first run

    # --- 1. Prepare Blob ---
    spk_ref_blob = client.pack_file(str(SPK_REF_PATH))
    
    # --- 2. Configuration ---
    # Server-side Init Params
    base_init = {
        "model_name": "F5TTS_v1_Base",
        "vocoder_name": "vocos", 
        "device": "cuda"
    }

    def run_case(name: str, text: str, kwargs: Dict[str, Any]):
        print(f"\n=== Running API Case: {name} ===")
        dest = OUT_DIR / f"{name}.wav"
        
        try:
            res = client.synth(
                adapter="f5tts",
                init=base_init,
                load_model={}, # No args needed for load_model logic
                clone_voice={
                    "ref_audio": spk_ref_blob,
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
        # --- Case 1: Standard ---
        run_case(
            "01_standard",
            "Hello! This is F5-TTS generating audio via the API.",
            {"speed": 1.0}
        )

        # --- Case 2: High Sampling Steps (Quality) ---
        run_case(
            "02_high_quality",
            "This generation uses more diffusion steps for higher fidelity.",
            {
                "nfe_step": 64, 
                "cfg_strength": 2.0
            }
        )

        # --- Case 3: Sway Sampling ---
        # Controls the flow matching sampling trajectory
        run_case(
            "03_sway_sampling",
            "Testing sway sampling coefficient variations.",
            {"sway_sampling_coef": -1.0}
        )

    finally:
        client.close()
        print("\nDone.")