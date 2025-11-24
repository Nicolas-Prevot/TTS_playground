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
    OUT_DIR = REPO_ROOT / "data" / "api_examples" / "vibevoice"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference audio
    SPK_REF_PATH = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    FR_REF_PATH = REPO_ROOT / "data" / "ref" / "fr" / "Ellie_Bishop_fr.wav"

    if not SPK_REF_PATH.exists():
        print(f"ERROR: Reference file not found: {SPK_REF_PATH}")
        sys.exit(1)

    # Initialize Client
    print(f"Connecting to {BASE_URL}...")
    client = TTSClient(BASE_URL, timeout=300.0) 

    # --- 1. Prepare Blobs ---
    en_ref_blob = client.pack_file(str(SPK_REF_PATH))
    
    fr_ref_blob = None
    if FR_REF_PATH.exists():
        fr_ref_blob = client.pack_file(str(FR_REF_PATH))

    # --- 2. Configuration ---
    base_init = {
        "model_id": "vibevoice/VibeVoice-7B",
        "device": "cuda",
        "cfg_scale": 1.3,
        "ddpm_steps": 10
    }

    def run_case(
        name: str, 
        text: str, 
        kwargs: Dict[str, Any], 
        clone_args: Dict[str, Any] = None
    ):
        print(f"\n=== Running API Case: {name} ===")
        dest = OUT_DIR / f"{name}.wav"
        
        try:
            res = client.synth(
                adapter="vibevoicetts",
                init=base_init,
                load_model={},
                clone_voice=clone_args or {"ref_audio": en_ref_blob},
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
        # --- Case 1: Standard Cloning ---
        run_case(
            "01_single_clone",
            "Hello! This is VibeVoice generated via the Playground API.",
            {"generation_config": {"do_sample": False}}
        )

        # --- Case 2: Creative Sampling ---
        run_case(
            "02_sampling",
            "This text is generated with higher temperature for more variation.",
            {"generation_config": {"do_sample": True, "temperature": 0.9}}
        )

        # --- Case 3: Multi-Speaker ---
        if fr_ref_blob:
            # Map speaker IDs to blob objects
            # Note: 'pack_speaker_map' helper in client can simplify this, 
            # but here we construct the dict manually to show structure.
            spk_map = {
                "1": en_ref_blob,
                "2": fr_ref_blob
            }
            
            script = (
                "Speaker 1: Welcome to the multi-speaker test.\n"
                "Speaker 2: Merci beaucoup. C'est un plaisir.\n"
                "Speaker 1: Excellent."
            )
            
            run_case(
                "03_multi_speaker",
                script,
                {"cfg_scale": 1.2},
                clone_args={"speaker_voices": spk_map}
            )

    finally:
        client.close()
        print("\nDone.")