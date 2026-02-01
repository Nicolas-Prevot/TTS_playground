import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "src"))

from tts_playground.client.tts_client import TTSClient


if __name__ == "__main__":
    BASE_URL = "http://localhost:7000"
    OUT_DIR = REPO_ROOT / "data" / "api_examples" / "kokoro"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {BASE_URL}...")
    client = TTSClient(BASE_URL, timeout=180.0)

    base_init = {
        "lang_code": "a",
        "voice": "af_heart",
    }

    def run_case(name: str, text: str, voice_id: str, lang: str, speed: float = 1.0):
        print(f"\n=== Running API Case: {name} ===")
        dest = OUT_DIR / f"{name}.wav"

        try:
            res = client.synth(
                adapter="kokoro",
                init=base_init,
                load_model={},
                clone_voice={"voice": voice_id, "lang_code": lang},
                synthesize={"text": text, "kwargs": {"speed": speed, "split_pattern": r"\n+"}},
                dest_path=str(dest),
                wait=True,
                download=True,
                timeout=600.0,
            )

            if res.get("state") == "SUCCESS":
                file_size = dest.stat().st_size if dest.exists() else 0
                print(f"[{name}] SUCCESS -> {dest.name} ({file_size} bytes)")
            else:
                print(f"[{name}] FAILED: {res.get('error')}")

        except Exception as e:
            print(f"[{name}] ERROR: {str(e)}")

    try:
        run_case(
            "01_us_bella",
            "I don't really care what you call me. I've been a silent spectator.",
            voice_id="af_bella",
            lang="a",
            speed=1.0,
        )

        run_case(
            "02_uk_george_fast",
            "Empires rise and fall. But always remember, I am mighty and enduring.",
            voice_id="bm_george",
            lang="b",
            speed=1.2,
        )

        run_case(
            "03_french",
            "Bonjour, ceci est un test de synthèse vocale avec Kokoro.",
            voice_id="ff_siwis",
            lang="f",
            speed=1.0,
        )

    finally:
        client.close()
        print("\nDone.")
