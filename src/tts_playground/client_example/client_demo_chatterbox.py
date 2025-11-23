from pathlib import Path
from typing import Any, Dict, Optional
import base64

from clienttts import TTSClient


if __name__ == "__main__":
    BASE_URL = "http://localhost:7000"
    OUT_DIR = Path("data/gen_api")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference audio for zero-shot cloning
    REF_AUDIO = "data/ref/basic_ref_en.wav"

    # Chatterbox is currently English-only
    TEXT = (
        "I don't really care what you call me. I've been a silent spectator, "
        "watching species evolve, empires rise and fall."
    )

    # Adapter init settings (optional defaults)
    base_init = {
        # "device": "cuda", 
    }

    client = TTSClient(BASE_URL, timeout=120.0)

    try:
        # Pack the reference audio file for upload
        ref_blob = client.pack_file(REF_AUDIO)

        def run_case(
            name: str,
            text: str,
            kwargs: Dict[str, Any],
            timeout_s: float = 600.0
        ):
            print(f"\n=== Running case: {name} ===")
            dest = OUT_DIR / f"chatterbox_{name}.wav"
            res = client.synth(
                adapter="chatterbox",
                init=base_init,
                load_model={},                  
                clone_voice={"ref_audio": ref_blob},
                synthesize={"text": text, "kwargs": kwargs},
                wait=True, download=True, timeout=timeout_s,
                dest_path=str(dest),
            )
            if res["state"] != "SUCCESS":
                print(f"[{name}] FAILED: {res.get('error')}")
            else:
                size = len(base64.b64decode(res["wav_b64"]))
                print(f"[{name}] OK -> {dest} (sr={res['sr']}, bytes={size})")

        # 1) Neutral / Standard
        run_case(
            "neutral",
            TEXT,
            kwargs={
                "exaggeration": 0.0,
                "cfg_weight": 0.5,
                "temperature": 0.8,
            },
        )

        # 2) High Exaggeration (More Emotional)
        run_case(
            "emotional",
            "But always remember, I am mighty and enduring!",
            kwargs={
                "exaggeration": 0.8,  # Higher value = more expressive
                "cfg_weight": 0.6,
                "temperature": 0.9,
            },
        )

        print("\nAll Chatterbox cases finished. Check files in:", OUT_DIR)

    finally:
        client.close()