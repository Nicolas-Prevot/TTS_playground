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
    OUT_DIR = REPO_ROOT / "data" / "api_examples" / "kyutai"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {BASE_URL}...")
    client = TTSClient(BASE_URL, timeout=300.0)

    # --- Configuration ---
    # Init parameters matching Kyutai adapter defaults
    base_init = {
        "hf_repo": "kyutai/tts-1.6b-en_fr",
        "n_q": 32,
        "temp": 0.6,
        "cfg_coef": 1.0,
        "device": "cuda"
    }

    def run_case(
        name: str, 
        text: str, 
        voice_sample: str
    ):
        print(f"\n=== Running API Case: {name} ===")
        dest = OUT_DIR / f"{name}.wav"
        
        try:
            # Note regarding 'clone_voice':
            # Kyutai adapter expects a 'voice_sample' string which corresponds 
            # to a preset ID in the model's repo (e.g. vctk/...).
            # We pass this directly in the dictionary.
            
            res = client.synth(
                adapter="kyutai",
                init=base_init,
                load_model={},
                clone_voice={
                    "voice_sample": voice_sample
                },
                synthesize={
                    "text": text, 
                    "kwargs": {}
                },
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
        # --- Case 1: English (VCTK) ---
        run_case(
            "01_english_vctk",
            "I don't really care what you call me. I've been a silent spectator.",
            voice_sample="vctk/p225_023.wav"
        )

        # --- Case 2: French (CML) ---
        run_case(
            "02_french_cml",
            "Bonjour! Ceci est une démonstration de synthèse vocale avec Kyutai.",
            voice_sample="cml-tts/fr/4724_3731_000031-0001.wav"
        )

        # --- Case 3: Expressive English (Expresso) ---
        # Using an Expresso dataset voice often yields more conversational tone
        run_case(
            "03_english_expresso",
            "Hey, did you see that? That was absolutely incredible!",
            voice_sample="expresso/ex01-ex02_default_001_channel1_168s.wav"
        )

    finally:
        client.close()
        print("\nDone.")