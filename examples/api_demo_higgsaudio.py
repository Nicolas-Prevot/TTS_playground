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
    
    OUT_DIR = REPO_ROOT / "data" / "api_examples" / "higgsaudio"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference
    SPK_REF_PATH = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    SPK_REF_TEXT = "Some call me nature, others call me mother nature."
    SCENE_DESC = "A clear voice speaking in a quiet room."

    if not SPK_REF_PATH.exists():
        print(f"ERROR: Reference file not found: {SPK_REF_PATH}")
        sys.exit(1)

    print(f"Connecting to {BASE_URL}...")
    client = TTSClient(BASE_URL, timeout=300.0)

    # --- 1. Prepare Blob ---
    spk_ref_blob = client.pack_file(str(SPK_REF_PATH))

    # --- 2. Configuration ---
    base_init = {
        "model_path": "bosonai/higgs-audio-v2-generation-3B-base",
        "audio_tokenizer_path": "bosonai/higgs-audio-v2-tokenizer",
        "use_static_kv_cache": True,
        "max_new_tokens": 4096,
        "device": "cuda"
    }

    def run_case(name: str, text: str, kwargs: Dict[str, Any]):
        print(f"\n=== Running API Case: {name} ===")
        dest = OUT_DIR / f"{name}.wav"
        
        try:
            res = client.synth(
                adapter="higgsaudio",
                init=base_init,
                load_model={},
                clone_voice={
                    "ref_audio": spk_ref_blob,
                    "ref_text": SPK_REF_TEXT,
                    "scene_prompt": SCENE_DESC
                },
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
        # --- Case 1: Standard Clone ---
        run_case(
            "01_standard",
            "This is a test of the isolated HiggsAudio adapter via API.",
            {"temperature": 0.95, "seed": 42}
        )

        # --- Case 2: Chunked Generation (Long text) ---
        long_text = (
            "We can break down longer texts into chunks to ensure stable generation "
            "over longer durations. The model processes the context and continues speaking."
        )
        run_case(
            "02_chunked",
            long_text,
            {
                "chunk_method": "word", 
                "chunk_max_word_num": 100,
                "generation_chunk_buffer_size": 2
            }
        )

    finally:
        client.close()
        print("\nDone.")