from pathlib import Path
from typing import Any, Dict, Optional
import base64

from clienttts import TTSClient


if __name__ == "__main__":
    BASE_URL = "http://localhost:7000"
    OUT_DIR = Path("data/gen_api")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference (must exist). Provide a *matching transcript* for best cloning.
    REF_AUDIO = "data/ref/basic_ref_en.wav"
    REF_TEXT  = "Some call me nature, others call me mother nature."
    SCENE_DESC = "A clear voice speaking in a quiet room."

    # --- Common init for HiggsAudio ---
    base_init = {
        "model_path": "bosonai/higgs-audio-v2-generation-3B-base",
        "audio_tokenizer_path": "bosonai/higgs-audio-v2-tokenizer",
        "device": None,                 # auto: cuda:0 if available, else cpu
        "use_static_kv_cache": True,    # speed-up on GPU
        "max_new_tokens": 4096,
        # (verbose is controlled by the adapter/logger; no extra flag here)
    }

    # Optional: a variant with static KV cache OFF (to A/B performance/behavior)
    base_init_no_cache = dict(base_init, use_static_kv_cache=False)

    # Example texts
    short_txt = "Hello! This is HiggsAudio via TTS_PLAYGROUND."
    long_txt = (
        "This is a longer passage intended to test chunked generation and buffering. "
        "We’ll split the text into word-based chunks so the model can process incrementally. "
        "The goal is smooth pacing without losing context across chunks."
    )
    zh_txt = "你好！这是一个中文的示例句子，用于测试分词、标点归一化和合成质量。"

    client = TTSClient(BASE_URL, timeout=120.0)

    try:
        ref_blob = client.pack_file(REF_AUDIO)

        def run_case(
            name: str,
            text: str,
            kwargs: Dict[str, Any],
            *,
            clone: Optional[Dict[str, Any]] = None,
            init_override: Optional[Dict[str, Any]] = None,
            timeout_s: float = 900.0,
        ):
            print(f"\n=== Running case: {name} ===")
            dest = OUT_DIR / f"higgsaudio_{name}.wav"
            res = client.synth(
                adapter="higgsaudio",       # ← change if your adapter uses a different name
                init=init_override or base_init,
                load_model={},              # no-op after first successful load with same init
                clone_voice=clone or {},    # sets/updates reference context
                synthesize={"text": text, "kwargs": kwargs},
                wait=True, download=True, timeout=timeout_s,
                dest_path=str(dest),
            )
            if res["state"] != "SUCCESS":
                print(f"[{name}] FAILED: {res.get('error')}")
            else:
                size = len(base64.b64decode(res["wav_b64"]))
                print(f"[{name}] OK → {dest}  (sr={res['sr']}, bytes={size})")

        # 1) Zero-shot baseline (no cloning, rely on default voice)
        run_case(
            "zeroshot_baseline",
            short_txt,
            kwargs={
                "temperature": 0.9,
                "top_p": 0.95,
                "seed": 123,  # for reproducibility
            },
            clone={},  # no ref audio/text
        )

        # 2) One-shot cloning with transcript + scene prompt
        run_case(
            "oneshot_clone_scene",
            "Hello world! This audio was generated using a cloned voice.",
            kwargs={
                "temperature": 0.95,
                "top_p": 0.9,
                "seed": 42,
            },
            clone={
                "ref_audio": ref_blob,
                "ref_text": REF_TEXT,
                "scene_prompt": SCENE_DESC,
            },
        )

        # 3) Long text with word-based chunking + small buffer window
        run_case(
            "long_chunked_wordbuf",
            long_txt,
            kwargs={
                "chunk_method": "word",
                "chunk_max_word_num": 120,         # ~120 "words" per chunk (Chinese uses jieba)
                "generation_chunk_buffer_size": 2, # keep only last 2 generated chunks in context
                "temperature": 0.95,
                "top_p": 0.95,
                "seed": 777,
            },
        )

        # 4) RAS window tweak (can affect rhythm/variability)
        run_case(
            "ras_window_tight",
            "Testing RAS settings for timing and repetition control.",
            kwargs={
                "ras_win_len": 5,
                "ras_win_max_num_repeat": 1,
                "temperature": 0.9,
                "top_p": 0.9,
                "seed": 31415,
            },
            clone={"ref_audio": ref_blob, "ref_text": REF_TEXT, "scene_prompt": SCENE_DESC},
        )

        # 5) Spicier sampling
        run_case(
            "sampling_spicy",
            "Let’s try a more expressive delivery with higher temperature.",
            kwargs={
                "temperature": 1.1,
                "top_k": 70,
                "top_p": 0.98,
                "seed": 2025,
            },
            clone={"ref_audio": ref_blob, "ref_text": REF_TEXT},
        )

        # 6) Chinese sample (punctuation is normalized automatically inside the adapter)
        run_case(
            "multilingual_zh",
            zh_txt,
            kwargs={
                "temperature": 0.95,
                "top_p": 0.95,
                "seed": 88,
                "chunk_method": "word",
                "chunk_max_word_num": 80,
            },
        )

        # 7) Same as #2 but with static KV cache disabled (A/B performance/latency)
        run_case(
            "oneshot_no_kv_cache",
            "This run disables the static KV cache to compare behavior and speed.",
            kwargs={"temperature": 0.95, "top_p": 0.9, "seed": 4242},
            clone={"ref_audio": ref_blob, "ref_text": REF_TEXT, "scene_prompt": SCENE_DESC},
            init_override=base_init_no_cache,
        )

        print("\nAll HiggsAudio cases finished. Check files in:", OUT_DIR)

    finally:
        client.close()
