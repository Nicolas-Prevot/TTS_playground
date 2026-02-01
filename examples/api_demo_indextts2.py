import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "src"))

from tts_playground.client.tts_client import TTSClient

if __name__ == "__main__":
    BASE_URL = "http://localhost:7000"

    OUT_DIR = REPO_ROOT / "data" / "api_examples" / "indextts2"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    SPK_REF_PATH = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    EMO_REF_PATH = REPO_ROOT / "data" / "ref" / "emo_hate.wav"

    if not SPK_REF_PATH.exists():
        print(f"ERROR: Speaker reference file not found: {SPK_REF_PATH}")
        sys.exit(1)

    print(f"Connecting to {BASE_URL}...")
    client = TTSClient(BASE_URL, timeout=180.0)

    spk_ref_blob = client.pack_file(str(SPK_REF_PATH))

    emo_ref_blob = None
    if EMO_REF_PATH.exists():
        emo_ref_blob = client.pack_file(str(EMO_REF_PATH))
    else:
        print(f"WARNING: Emotion reference file not found: {EMO_REF_PATH}. Skipping emotion audio tests.")

    base_init = {
        "model_dir": "/workspace/checkpoints/indextts2",
        "cfg_path": "/workspace/checkpoints/indextts2/config.yaml",
        "use_fp16": False,
        "device": None,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.8,
        # upstream name:
        "max_text_tokens_per_sentence": 120,
    }

    def run_case(name: str, text: str, kwargs: Dict[str, Any]):
        print(f"\n=== Running API Case: {name} ===")
        dest = OUT_DIR / f"{name}.wav"

        try:
            res = client.synth(
                adapter="indextts2",
                init=base_init,
                load_model={},
                clone_voice={"ref_audio": spk_ref_blob},
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
            print(f"[{name}] ERROR: {str(e)}")

    try:
        run_case(
            "01_baseline",
            "Hello! This is IndexTTS2 generating audio via the API.",
            {"use_random": False},
        )

        if emo_ref_blob:
            run_case(
                "02_emo_audio",
                "I am extremely angry about this situation!",
                {
                    "emo_audio_prompt": emo_ref_blob,
                    "emo_alpha": 0.8,
                    "use_random": False,
                },
            )

        run_case(
            "03_emo_vector",
            "I'm feeling very calm and happy right now.",
            {
                "emo_vector": [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6],
                "emo_alpha": 1.0,
                "use_random": False,
            },
        )

        long_txt = (
            "This is a test of longer text synthesis through the API. "
            "The system should handle the segmentation and processing on the server side, "
            "returning a single stitched audio file."
        )
        run_case(
            "04_long_text",
            long_txt,
            {
                "max_text_tokens_per_sentence": 50,
                "interval_silence": 400,
            },
        )

    finally:
        client.close()
        print("\nDone.")
