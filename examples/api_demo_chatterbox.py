import sys
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

    REF_AUDIO = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    if not REF_AUDIO.exists():
        print(f"ERROR: {REF_AUDIO} not found")
        sys.exit(1)

    client = TTSClient(BASE_URL, timeout=180.0)

    base_init = {
        # "device": "cuda",
        "device": "cpu",
    }

    def run_case(name: str, text: str, clone_args: Dict[str, Any], kwargs: Dict[str, Any]):
        print(f"\n=== Running case: {name} ===")
        dest = OUT_DIR / f"{name}.wav"

        try:
            res = client.synth(
                adapter="chatterbox",
                init=base_init,
                load_model={},
                clone_voice=clone_args,
                synthesize={"text": text, "kwargs": kwargs},
                wait=True,
                download=True,
                timeout=600.0,
                dest_path=str(dest),
            )

            if res.get("state") == "SUCCESS":
                size = dest.stat().st_size if dest.exists() else 0
                print(f"[{name}] SUCCESS -> {dest.name} ({size} bytes)")
            else:
                print(f"[{name}] FAILED: {res.get('error')}")
        except Exception as e:
            print(f"[{name}] ERROR: {e}")

    try:
        ref_blob = client.pack_file(str(REF_AUDIO))

        # 1) First call: clone voice (uploads ref audio)
        run_case(
            "01_clone_then_neutral",
            "I don't really care what you call me. I've been a silent spectator, watching species evolve.",
            clone_args={"ref_audio": ref_blob},
            kwargs={"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},
        )

        # 2) Second call: reuse cached voice conditionals in the runner (NO ref audio)
        run_case(
            "02_reuse_cached_voice",
            "But always remember, I am mighty and enduring!",
            clone_args={},  # IMPORTANT: empty dict => runner does NOT call clone_voice()
            kwargs={"exaggeration": 0.85, "cfg_weight": 0.4, "temperature": 0.9},
        )

    finally:
        client.close()
        print("\nDone.")
