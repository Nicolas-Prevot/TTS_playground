# client_demo_openaudio_s1mini.py
from pathlib import Path
from typing import Any, Dict, Optional
import base64

from clienttts import TTSClient

if __name__ == "__main__":
    BASE_URL = "http://localhost:7000"
    OUT_DIR = Path("data/gen_api")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference audio + its transcript (helps conditioning)
    SPK_REF_PATH = "data/ref/basic_ref_en.wav"
    SPK_REF_TEXT = "Some call me nature, others call me mother nature."

    # Init for OpenAudio S1 Mini (omit 'device' to let the adapter decide)
    base_init = {
        "llama_checkpoint_dir": "checkpoints/openaudio-s1-mini",
        "codec_checkpoint_path": "checkpoints/openaudio-s1-mini/codec.pth",
        "decoder_config_name": "modded_dac_vq",
        "half": False,        # True = torch.float16 for the semantic model
        "compile": False,     # set True if you’ve tested torch.compile on your setup
    }

    # Texts
    short_txt = "Hello! This is OpenAudio S1 Mini via TTS_PLAYGROUND."
    long_txt = (
        "This is a longer passage to hear how the model maintains coherence across multiple sentences. "
        "We will also test iterative prompting with chunked generation."
    )

    client = TTSClient(BASE_URL, timeout=120.0)

    try:
        en_ref = client.pack_file(SPK_REF_PATH)

        def run_case(
            name: str,
            text: str,
            kwargs: Dict[str, Any],
            timeout_s: float = 600.0,
            init_override: Optional[Dict[str, Any]] = None,
        ):
            print(f"\n=== Running case: {name} ===")
            dest = OUT_DIR / f"openaudio_s1mini_{name}.wav"
            res = client.synth(
                adapter="openaudios1mini",
                init=init_override or base_init,
                load_model={},                     # no-op after the first load
                clone_voice={"ref_audio": en_ref, "ref_text": SPK_REF_TEXT},
                synthesize={"text": text, "kwargs": kwargs},
                wait=True, download=True, timeout=timeout_s,
                dest_path=str(dest),
            )
            if res["state"] != "SUCCESS":
                print(f"[{name}] FAILED: {res.get('error')}")
            else:
                size = len(base64.b64decode(res["wav_b64"]))
                print(f"[{name}] OK → {dest}  (sr={res['sr']}, bytes={size})")

        # 1) Baseline (defaults from adapter)
        run_case(
            "baseline",
            short_txt,
            {
                # adapter defaults shown explicitly for clarity
                "top_p": 0.8,
                "temperature": 0.8,
                "repetition_penalty": 1.1,
                "max_new_tokens": 0,   # 0 = let model decide
            },
        )

        # 2) Calmer / safer sampling
        run_case(
            "safer_sampling",
            short_txt,
            {
                "temperature": 0.7,
                "top_p": 0.75,
                "repetition_penalty": 1.15,
                "max_new_tokens": 0,
                "seed": 1234,
            },
        )

        # 3) More expressive
        run_case(
            "expressive",
            short_txt,
            {
                "temperature": 1.0,
                "top_p": 0.95,
                "repetition_penalty": 1.1,
                "max_new_tokens": 0,
                "seed": 4242,
            },
        )

        # 4) Long text with iterative prompting (chunked semantic generation)
        #    Increase chunk_length if you want longer segments per iteration.
        run_case(
            "iterative_chunk_160",
            long_txt,
            {
                "chunk_length": 160,    # semantic tokens per step
                "temperature": 0.85,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "max_new_tokens": 0,
            },
        )

        # 5) Stronger anti-repetition (useful on long inputs)
        run_case(
            "anti_repeat",
            long_txt,
            {
                "temperature": 0.9,
                "top_p": 0.9,
                "repetition_penalty": 1.35,
                "max_new_tokens": 0,
                "seed": 777,
            },
        )

        # 6) Deterministic-ish A/B (same seed + params → stable prosody)
        run_case(
            "deterministic_seed",
            short_txt,
            {
                "temperature": 0.8,
                "top_p": 0.85,
                "repetition_penalty": 1.1,
                "max_new_tokens": 0,
                "seed": 2024,
            },
        )

        print("\nAll OpenAudio S1 Mini cases finished. Compare the resulting WAVs in:", OUT_DIR)

    finally:
        client.close()
