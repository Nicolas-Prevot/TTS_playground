# client_demo_vibevoice.py
from pathlib import Path
from typing import Any, Dict, Optional
import base64

from clienttts import TTSClient


if __name__ == "__main__":
    BASE_URL = "http://localhost:7000"
    OUT_DIR = Path("data/gen_api")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference audios (must exist)
    EN_REF_PATH = "data/ref/basic_ref_en.wav"
    FR_REF_PATH = "data/ref/fr/Ellie_Bishop_fr.wav"   # used in multi-speaker demos

    # Common init for VibeVoice (mirrors adapter defaults; adjust as needed)
    base_init = {
        "model_id": "vibevoice/VibeVoice-7B",  # or "vibevoice/VibeVoice-1.5B"
        "device": None,                        # auto: cuda > mps > cpu
        "torch_dtype": None,                   # auto: bf16 on CUDA, f32 otherwise
        "attn_implementation": None,           # auto: flash_attention_2 on CUDA else sdpa
        "cfg_scale": 1.3,                      # classifier-free guidance
        "ddpm_steps": 10,                      # diffusion steps
        "is_prefill": True,                    # enable cloning when refs are present
        "generation_config": {"do_sample": False},  # deterministic baseline
        # "lora_checkpoint": "/path/to/lora",   # optional LoRA
        # "verbose": True,
    }

    # Example texts
    single_txt = "Hello! This is VibeVoice via TTS_PLAYGROUND."
    dialog_txt = (
        "Speaker 1: Hi! I'm the first speaker.\n"
        "Speaker 2: And I'm the second speaker.\n"
        "Speaker 1: Great to meet you!"
    )

    client = TTSClient(BASE_URL, timeout=120.0)

    try:
        en_ref = client.pack_file(EN_REF_PATH)
        fr_ref = client.pack_file(FR_REF_PATH)

        def run_case(
            name: str,
            text: str,
            kwargs: Dict[str, Any],
            *,
            clone: Optional[Dict[str, Any]] = None,
            timeout_s: float = 600.0,
        ):
            print(f"\n=== Running case: {name} ===")
            dest = OUT_DIR / f"vibevoice_{name}.wav"
            res = client.synth(
                adapter="vibevoicetts",
                init=base_init,
                load_model={},                         # no-op after first successful load
                clone_voice=clone or {},               # sticky clone config
                synthesize={"text": text, "kwargs": kwargs},
                wait=True, download=True, timeout=timeout_s,
                dest_path=str(dest),
            )
            if res["state"] != "SUCCESS":
                print(f"[{name}] FAILED: {res.get('error')}")
            else:
                size = len(base64.b64decode(res["wav_b64"]))
                print(f"[{name}] OK → {dest}  (sr={res['sr']}, bytes={size})")

        # ------------------------------------------------------------------
        # 1) Single-speaker with voice cloning (prefill ON)
        #    Uses one reference voice; deterministic generation.
        run_case(
            "single_prefill_on",
            single_txt,
            kwargs={"generation_config": {"do_sample": False}},
            clone={"ref_audio": en_ref},
        )

        # 2) Single-speaker with prefill OFF (same text, no cloning)
        #    Good A/B against the previous file.
        run_case(
            "single_prefill_off",
            single_txt,
            kwargs={
                "is_prefill": False,                         # disable cloning even if ref exists
                "generation_config": {"do_sample": False},   # deterministic
            },
            clone={"ref_audio": en_ref},  # still send the ref to keep setup identical
        )

        # 3) Single-speaker with mild sampling (still no cloning to isolate sampling effect)
        run_case(
            "single_sampling_mild",
            single_txt,
            kwargs={
                "is_prefill": False,
                "generation_config": {"do_sample": True, "temperature": 0.8, "top_p": 0.9},
                "seed": 42,  # make it reproducible
            },
            clone={"ref_audio": en_ref},
        )

        # 4) Multi-speaker using an explicit speaker→voice map (sticky clone)
        spk_map_blob = client.pack_speaker_map({"1": EN_REF_PATH, "2": FR_REF_PATH})
        run_case(
            "multi_map",
            dialog_txt,
            kwargs={"cfg_scale": 1.2},  # slightly more guidance
            clone={"speaker_voices": spk_map_blob},
        )

        # 5) Multi-speaker using an ordered voice list (Speaker 1 -> list[0], etc.)
        run_case(
            "multi_list",
            dialog_txt,
            kwargs={},  # defaults
            clone={"voice_samples": [en_ref, fr_ref]},
        )

        # 6) Per-call override of speaker_voices (no sticky clone)
        #    Useful when you don’t want to change the process-global clone state.
        spk_map_inline = client.pack_speaker_map({"1": en_ref, "2": fr_ref})
        run_case(
            "multi_per_call_override",
            dialog_txt,
            kwargs={"speaker_voices": spk_map_inline, "cfg_scale": 1.3},
            clone={},  # nothing sticky
        )

        # 7) Expressive settings: stronger guidance + more DDPM steps + sampling
        run_case(
            "expressive_cfg_steps_sampling",
            single_txt,
            kwargs={
                "cfg_scale": 1.6,
                "ddpm_steps": 16,
                "generation_config": {"do_sample": True, "temperature": 0.95, "top_p": 0.95},
                "seed": 1234,  # set or remove to taste
            },
            clone={"ref_audio": en_ref},
        )

        # 8) Repro test: same seed/params should match (except for nondeterminism on some backends)
        run_case(
            "repro_seed_A",
            single_txt,
            kwargs={"generation_config": {"do_sample": True, "temperature": 0.9, "top_p": 0.9}, "seed": 777},
            clone={"ref_audio": en_ref},
        )
        run_case(
            "repro_seed_B",
            single_txt,
            kwargs={"generation_config": {"do_sample": True, "temperature": 0.9, "top_p": 0.9}, "seed": 777},
            clone={"ref_audio": en_ref},
        )

        print("\nAll VibeVoice cases finished. Check files in:", OUT_DIR)

    finally:
        client.close()
