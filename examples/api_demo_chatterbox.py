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
    OUT_DIR = REPO_ROOT / "data" / "api_examples" / "chatterbox"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference audio
    REF_AUDIO = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    if not REF_AUDIO.exists():
        print(f"ERROR: {REF_AUDIO} not found")
        sys.exit(1)

    # Chatterbox is currently English-only
    TEXT = (
        "I don't really care what you call me. I've been a silent spectator, "
        "watching species evolve, empires rise and fall."
    )

    # Adapter init settings
    base_init = {
        # "device": "cuda", 
        "device": "cpu",
    }

    client = TTSClient(BASE_URL, timeout=120.0)

    try:
        print(f"Connecting to {BASE_URL}...")
        ref_blob = client.pack_file(str(REF_AUDIO))

        def run_case(
            name: str,
            text: str,
            kwargs: Dict[str, Any],
            timeout_s: float = 600.0
        ):
            print(f"\n=== Running case: {name} ===")
            dest = OUT_DIR / f"{name}.wav"
            
            try:
                res = client.synth(
                    adapter="chatterbox",
                    init=base_init,
                    load_model={},                  
                    clone_voice={"ref_audio": ref_blob},
                    synthesize={"text": text, "kwargs": kwargs},
                    wait=True, 
                    download=True, 
                    timeout=timeout_s,
                    dest_path=str(dest),
                )
                if res["state"] != "SUCCESS":
                    print(f"[{name}] FAILED: {res.get('error')}")
                else:
                    file_size = dest.stat().st_size if dest.exists() else 0
                    print(f"[{name}] SUCCESS -> {dest.name} ({file_size} bytes)")
            except Exception as e:
                print(f"[{name}] ERROR: {str(e)}")

        # 1) Neutral
        run_case(
            "01_neutral",
            TEXT,
            kwargs={
                "exaggeration": 0.5,
                "cfg_weight": 0.5,
                "temperature": 0.8,
            },
        )

        # 2) Emotional / Dramatic
        run_case(
            "02_emotional",
            "But always remember, I am mighty and enduring!",
            kwargs={
                "exaggeration": 0.85,
                "cfg_weight": 0.4, # Lower CFG helps pacing with high emotion
                "temperature": 0.9,
            },
        )

    finally:
        client.close()
        print("\nDone.")