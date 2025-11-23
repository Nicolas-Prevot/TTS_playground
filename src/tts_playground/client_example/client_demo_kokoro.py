from pathlib import Path
from typing import Any, Dict
import base64

from clienttts import TTSClient


if __name__ == "__main__":
    BASE_URL = "http://localhost:7000"
    OUT_DIR = Path("data/gen_api")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Kokoro uses built-in voice IDs (strings), not file uploads for cloning.
    # See KokoroTTS.md for voice list.

    # Adapter Init
    base_init = {
        "lang_code": "a",       # Default to American English
        "voice": "af_heart",    # Default voice
    }

    client = TTSClient(BASE_URL, timeout=60.0)

    try:
        def run_case(
            name: str,
            text: str,
            voice: str,
            lang_code: str,
            speed: float = 1.0
        ):
            print(f"\n=== Running case: {name} ===")
            dest = OUT_DIR / f"kokoro_{name}.wav"
            
            res = client.synth(
                adapter="kokoro",
                init=base_init,
                load_model={},
                # Kokoro's clone_voice takes voice ID and lang code string, not a file
                clone_voice={"voice": voice, "lang_code": lang_code},
                synthesize={"text": text, "kwargs": {"speed": speed}},
                wait=True, download=True, timeout=300,
                dest_path=str(dest),
            )
            
            if res["state"] != "SUCCESS":
                print(f"[{name}] FAILED: {res.get('error')}")
            else:
                size = len(base64.b64decode(res["wav_b64"]))
                print(f"[{name}] OK -> {dest} (sr={res['sr']}, bytes={size})")

        # 1) American English (Bella) - Standard speed
        run_case(
            "us_female_bella",
            "I don't really care what you call me. I've been a silent spectator.",
            voice="af_bella",
            lang_code="a",
            speed=1.0
        )

        # 2) British Male (George) - Slower speed
        run_case(
            "uk_male_george",
            "Empires rise and fall. But always remember, I am mighty and enduring.",
            voice="bm_george",
            lang_code="b",
            speed=0.85
        )

        # 3) French (Siwis)
        run_case(
            "french_siwis",
            "À l'époque classique, à Athènes, les auteurs doivent présenter au concours trois tragédies.",
            voice="ff_siwis",
            lang_code="f",
            speed=1.0
        )

        print("\nAll Kokoro cases finished. Check files in:", OUT_DIR)

    finally:
        client.close()