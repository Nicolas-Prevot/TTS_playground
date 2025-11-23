from pathlib import Path
from typing import Any, Dict
import base64

from clienttts import TTSClient


if __name__ == "__main__":
    BASE_URL = "http://localhost:7000"
    OUT_DIR = Path("data/gen_api")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Kyutai uses preset voice paths/IDs string. 
    # We pass these strings directly instead of packing a file blob.
    
    # Init parameters matching Kyutai adapter
    base_init = {
        "hf_repo": "kyutai/tts-1.6b-en_fr",
        "n_q": 32,
        "temp": 0.6,
        "cfg_coef": 1.0,
        "device": None # auto
    }

    client = TTSClient(BASE_URL, timeout=120.0)

    try:
        def run_case(
            name: str,
            text: str,
            voice_sample: str, # String ID relative to repo (e.g. "vctk/p225_023.wav")
        ):
            print(f"\n=== Running case: {name} ===")
            dest = OUT_DIR / f"kyutai_{name}.wav"
            
            res = client.synth(
                adapter="kyutai",
                init=base_init,
                load_model={},
                # clone_voice takes "voice_sample" string
                clone_voice={"voice_sample": voice_sample},
                synthesize={"text": text, "kwargs": {}},
                wait=True, download=True, timeout=300,
                dest_path=str(dest),
            )
            
            if res["state"] != "SUCCESS":
                print(f"[{name}] FAILED: {res.get('error')}")
            else:
                size = len(base64.b64decode(res["wav_b64"]))
                print(f"[{name}] OK -> {dest} (sr={res['sr']}, bytes={size})")

        # 1) English VCTK Voice
        run_case(
            "en_vctk",
            "I don't really care what you call me. I've been a silent spectator.",
            voice_sample="vctk/p225_023.wav"
        )

        # 2) French CML-TTS Voice
        run_case(
            "fr_cml",
            "À l'époque classique, à Athènes, les auteurs doivent présenter au concours trois tragédies.",
            voice_sample="cml-tts/fr/4724_3731_000031-0001.wav"
        )

        print("\nAll Kyutai cases finished. Check files in:", OUT_DIR)

    finally:
        client.close()