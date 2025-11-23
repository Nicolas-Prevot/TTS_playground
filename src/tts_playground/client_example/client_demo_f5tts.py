from pathlib import Path
from typing import Any, Dict, Optional
import base64

from clienttts import TTSClient


if __name__ == "__main__":
    BASE_URL = "http://localhost:7000"
    OUT_DIR = Path("data/gen_api")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference cloning (must exist) — text should closely match the audio
    REF_AUDIO = "data/ref/basic_ref_en.wav"
    REF_TEXT  = "Some call me nature, others call me mother nature."

    # --- Init presets (pick VOCOS by default; add BigVGAN A/B case later) ---
    base_init_vocos = {
        "model_name": "F5TTS_v1_Base",   # supported with vocos
        "vocoder_name": "vocos",         # or "bigvgan" (see alt init below)
        # leave "device" unset to use the adapter's default (auto from F5 utils)
        # "hf_cache_dir": "hf_cache",    # optional
    }
    base_init_bigvgan = {
        "model_name": "F5TTS_Base",      # BigVGAN-compatible
        "vocoder_name": "bigvgan",
    }

    # Example texts
    short_txt = "Hello! This is F5-TTS via TTS_PLAYGROUND."
    long_txt = (
        "This is a longer passage to hear stability across sentences. "
        "We will try different diffusion steps, guidance, and cross-fade settings "
        "to smooth transitions and adjust overall loudness."
    )

    client = TTSClient(BASE_URL, timeout=120.0)

    try:
        ref_blob = client.pack_file(REF_AUDIO)

        def run_case(
            name: str,
            text: str,
            kwargs: Dict[str, Any],
            *,
            init_override: Optional[Dict[str, Any]] = None,
            timeout_s: float = 900.0,
        ):
            print(f"\n=== Running case: {name} ===")
            dest = OUT_DIR / f"f5tts_{name}.wav"
            res = client.synth(
                adapter="f5tts",                 # ← change if your adapter name differs
                init=init_override or base_init_vocos,
                load_model={},                   # no-op after first successful load with same init
                clone_voice={"ref_audio": ref_blob, "ref_text": REF_TEXT},
                synthesize={"text": text, "kwargs": kwargs},
                wait=True, download=True, timeout=timeout_s,
                dest_path=str(dest),
            )
            if res["state"] != "SUCCESS":
                print(f"[{name}] FAILED: {res.get('error')}")
            else:
                size = len(base64.b64decode(res["wav_b64"]))
                print(f"[{name}] OK → {dest}  (sr={res['sr']}, bytes={size})")

        # 1) Baseline (Vocos + default sampler)
        run_case(
            "baseline_vocos",
            short_txt,
            kwargs={
                "speed": 1.0,
                # defaults for: nfe_step, cfg_strength, sway_sampling_coef, etc.
            },
        )

        # 2) Slower / faster speaking rate
        run_case("speed_085", short_txt, kwargs={"speed": 0.85})
        run_case("speed_120", short_txt, kwargs={"speed": 1.20})

        # 3) Fewer vs more diffusion steps (quality vs speed tradeoff)
        run_case("nfe_8_fast",  short_txt, kwargs={"nfe_step": 8})
        run_case("nfe_32_clean", short_txt, kwargs={"nfe_step": 32})

        # 4) Stronger guidance (more faithful/less diverse) + mild sway sampling
        run_case(
            "cfg_strong_sway_mild",
            short_txt,
            kwargs={"cfg_strength": 2.0, "sway_sampling_coef": 0.2}
        )

        # 5) Long text with cross-fade smoothing across segments (+ slightly lower loudness)
        run_case(
            "long_crossfade_quiet",
            long_txt,
            kwargs={
                "nfe_step": 24,
                "cfg_strength": 1.6,
                "cross_fade_duration": 0.18,   # seconds
                "target_rms": 0.08,            # lower = quieter output
            },
        )

        # 6) Constrain duration (aim for a specific length); useful for timed reads
        run_case(
            "fixed_duration_3_5s",
            "Timing test. This line is constrained to roughly three and a half seconds.",
            kwargs={
                "fix_duration": 3.5,   # seconds (adapter forwards to F5 infer)
                "nfe_step": 20,
                "cfg_strength": 1.4,
            },
        )

        # 7) BigVGAN A/B (model/vocoder combo that supports it)
        run_case(
            "bigvgan_ab",
            "BigVGAN comparison run for timbre/brightness differences.",
            kwargs={"nfe_step": 24, "cfg_strength": 1.5},
            init_override=base_init_bigvgan,
        )

        print("\nAll F5-TTS cases finished. Check files in:", OUT_DIR)

    finally:
        client.close()
