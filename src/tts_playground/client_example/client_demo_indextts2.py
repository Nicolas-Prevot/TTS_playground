from pathlib import Path
from typing import Any, Dict, Optional
import base64

from clienttts import TTSClient

if __name__ == "__main__":
    BASE_URL = "http://localhost:7000"
    OUT_DIR = Path("data/gen_api")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference audios (must exist)
    SPK_REF_PATH = "data/ref/basic_ref_en.wav"
    EMO_REF_PATH = "data/ref/emo_hate.wav"   # a very different emotion helps hear contrast

    # Common init for IndexTTS2
    base_init = {
        "model_dir": "checkpoints/indextts2",
        "cfg_path": "checkpoints/indextts2/config.yaml",
        "use_fp16": False,
        "device": None,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.8,
        "max_text_tokens_per_segment": 120,
        # "verbose": True,   # uncomment if you want extra logs from the adapter/model
    }

    # Two example texts: one short, one multi-sentence
    short_txt = "Hello! This is IndexTTS2 via TTS_PLAYGROUND."
    long_txt = (
        "This is a slightly longer passage designed to cross a few sentence boundaries. "
        "It helps us hear how the model handles segmentation, pacing, and the inter-segment silence. "
        "We will also vary sampling parameters to hear more expressive delivery."
    )

    # --------------------------------------------------------------------------
    client = TTSClient(BASE_URL, timeout=60.0)

    try:
        en_ref = client.pack_file(SPK_REF_PATH)
        emo_ref = client.pack_file(EMO_REF_PATH)

        def run_case(name: str, text: str, kwargs: Dict[str, Any], timeout_s: float = 600.0):
            print(f"\n=== Running case: {name} ===")
            dest = OUT_DIR / f"indextts2_{name}.wav"
            res = client.synth(
                adapter="indextts2",
                init=base_init,
                load_model={},                  # load request (no-ops after first successful load)
                clone_voice={"ref_audio": en_ref},
                synthesize={"text": text, "kwargs": kwargs},
                wait=True, download=True, timeout=timeout_s,
                dest_path=str(dest),
            )
            if res["state"] != "SUCCESS":
                print(f"[{name}] FAILED: {res.get('error')}")
            else:
                print(f"[{name}] OK → {dest}  (sr={res['sr']}, bytes={len(base64.b64decode(res['wav_b64']))})")

        # 1) Baseline: emotion from speaker voice (no emo_* passed)
        run_case(
            "baseline_speaker_emotion",
            short_txt,
            {
                "use_random": False,           # remove sampling noise for easier A/B
            },
        )

        # 2) Emotion from separate audio, moderate alpha (should be audible but subtle)
        run_case(
            "emo_audio_prompt_alpha07",
            short_txt,
            {
                "emo_audio_prompt": emo_ref,   # blob gets staged to a file by the runner
                "emo_alpha": 0.7,
                "use_random": False,
            },
        )

        # 3) Emotion from separate audio, strong alpha (max effect)
        run_case(
            "emo_audio_prompt_alpha10",
            short_txt,
            {
                "emo_audio_prompt": emo_ref,
                "emo_alpha": 1.0,
                "use_random": False,
            },
        )

        # 4) Emotion from vector (happy + calm), deterministic
        run_case(
            "emo_vector_happy_calm",
            short_txt,
            {
                "emo_vector": [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6],
                "emo_alpha": 1.0,
                "use_random": False,
                # you can also tweak generation knobs here if you want:
                # "temperature": 0.7, "top_p": 0.92,
            },
        )

        # 5) Emotion from natural language text (QwenEmotion), moderate alpha
        run_case(
            "emo_text_melancholic",
            "A short melancholic passage, softly spoken.",
            {
                "use_emo_text": True,
                "emo_text": "gentle, slightly melancholic, intimate",
                "emo_alpha": 0.8,
                "use_random": False,
            },
        )

        # 6) Longer text with forced segmentation and longer pauses
        run_case(
            "segmentation_interval300_ms",
            long_txt,
            {
                "max_text_tokens_per_segment": 40,  # force smaller segments
                "interval_silence": 300,            # 300 ms pause between segments
                "use_random": False,
            },
        )

        # 7) More expressive sampling (to hear style variability)
        run_case(
            "expressive_sampling",
            long_txt,
            {
                "do_sample": True,
                "temperature": 0.95,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 8.0,
                # keep emotion from speaker so we isolate sampling effects
                "use_random": True,                 # allow model-side randomness
            },
        )

        print("\nAll cases finished. Compare the resulting WAVs in:", OUT_DIR)

    finally:
        client.close()