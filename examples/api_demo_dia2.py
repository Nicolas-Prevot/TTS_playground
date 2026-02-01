import sys
from pathlib import Path
from typing import Any, Dict

# Add the source root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "src"))

from tts_playground.client.tts_client import TTSClient


if __name__ == "__main__":
    BASE_URL = "http://localhost:7000"

    OUT_DIR = REPO_ROOT / "data" / "api_examples" / "dia2"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Optional: reuse shared reference audio for conditioning
    SPK_REF_PATH = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    if not SPK_REF_PATH.exists():
        print(f"WARNING: Reference file not found: {SPK_REF_PATH}")
        SPK_REF_PATH = None

    print(f"Connecting to {BASE_URL}...")
    client = TTSClient(BASE_URL, timeout=180.0)

    # --- 1. Prepare Blob (if ref exists) ---
    spk_ref_blob = client.pack_file(str(SPK_REF_PATH)) if SPK_REF_PATH else None

    # --- 2. Init settings (mirrors Dia2Adapter defaults) ---
    base_init = {
        "repo_id": "nari-labs/Dia2-2B",
        "device": "cuda",         # or "cpu" (slower)
        "dtype": "bfloat16",
        "cfg_scale": 6.0,         # matches upstream CLI quickstart
        "audio_temperature": 0.8,
        "audio_top_k": 50,
        "use_cuda_graph": True,
    }

    def run_case(
        name: str,
        text: str,
        kwargs: Dict[str, Any],
        use_prefix: bool = False,
    ):
        print(f"\n=== Running API Case: {name} ===")
        dest = OUT_DIR / f"{name}.wav"

        clone_args: Dict[str, Any] = {}
        if use_prefix and spk_ref_blob is not None:
            clone_args = {
                "prefix_speaker_1": spk_ref_blob,
                "include_prefix": False,  # conditioning only (do not prepend ref audio)
            }

        try:
            res = client.synth(
                adapter="dia2",
                init=base_init,
                load_model={},
                clone_voice=clone_args,
                synthesize={"text": text, "kwargs": kwargs},
                dest_path=str(dest),
                wait=True,
                download=True,
            )

            if res.get("state") == "SUCCESS":
                file_size = dest.stat().st_size if dest.exists() else 0
                print(f"[{name}] SUCCESS -> {dest.name} ({file_size} bytes)")
            else:
                print(f"[{name}] FAILED: {res.get('error')}")
        except Exception as e:
            print(f"[{name}] ERROR: {e}")

    try:
        # --- Case 1: Simple dialogue, no prefix audio ---
        script1 = (
            "[S1] Hello! This is Dia2 generating audio via the Playground API.\n"
            "[S2] Great, everything is wired up correctly."
        )
        run_case(
            "01_basic",
            script1,
            kwargs={},
            use_prefix=False,
        )

        # --- Case 2: Conditioned on prefix_speaker_1 (if available) ---
        script2 = (
            "[S1] This line is conditioned on the prefix audio you provided.\n"
            "[S2] The voice should be closer to the reference speaker."
        )
        run_case(
            "02_with_prefix",
            script2,
            kwargs={},
            use_prefix=True,
        )

        # --- Case 3: Higher temperature and top_k for more variation ---
        script3 = (
            "[S1] Let us increase the sampling temperature and top k.\n"
            "[S2] The prosody should become a little more varied and playful."
        )
        run_case(
            "03_high_temp",
            script3,
            kwargs={
                "temperature": 0.95,
                "top_k": 80,
            },
            use_prefix=True,
        )

    finally:
        client.close()
        print("\nDone.")
