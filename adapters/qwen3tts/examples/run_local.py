import sys
from pathlib import Path

import torch
from loguru import logger

from tts_adapter_qwen3tts.adapter import Qwen3TTSAdapter


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent

    OUT_DIR = REPO_ROOT / "data" / "local_examples" / "qwen3tts"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    REF_AUDIO = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    # IMPORTANT: for ICL cloning, REF_TEXT must match REF_AUDIO closely.
    REF_TEXT = "Some call me nature, others call me mother nature."

    if not REF_AUDIO.exists():
        logger.error(f"Missing reference audio: {REF_AUDIO}")
        sys.exit(1)

    DEVICE = _default_device()
    DTYPE = "bfloat16" if DEVICE.startswith("cuda") else ("float16" if DEVICE == "mps" else "float32")
    ATTN = "sdpa"  # flash_attention_2 is optional; not installed by default in this adapter

    logger.info(f"Repo Root: {REPO_ROOT}")
    logger.info(f"Output Dir: {OUT_DIR}")
    logger.info(f"Device={DEVICE} dtype={DTYPE} attn_implementation={ATTN}")

    # -------------------------------------------------------------------------------------------
    # 1) CustomVoice (preset speakers)
    # -------------------------------------------------------------------------------------------
    logger.info("=== 1) CustomVoice ===")
    cv = Qwen3TTSAdapter(
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device=DEVICE,
        dtype=DTYPE,
        attn_implementation=ATTN,
    )
    cv.load_model()

    speakers = None
    if hasattr(cv.model, "get_supported_speakers"):
        speakers = list(cv.model.get_supported_speakers())
        logger.info(f"Supported speakers: {speakers}")

    # Generate one sample per speaker (9 speakers in the official release)
    if not speakers:
        speakers = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]

    for spk in speakers:
        out = OUT_DIR / f"01_customvoice_{spk}.wav"
        out.write_bytes(
            cv.synthesize(
                "Good evening. This is a preset speaker test for Qwen3-TTS CustomVoice.",
                language="English",
                speaker=spk,
                instruct="Calm, professional, slightly slower pace.",
            )
        )
        logger.success(f"Saved: {out.name}")

    # Style sweep on one speaker to show controllability
    style_sweep = [
        ("neutral", None),
        ("excited", "Energetic, enthusiastic, fast pace, lively intonation."),
        ("whisper", "Whispering, intimate, soft, slow and careful."),
    ]
    for tag, inst in style_sweep:
        out = OUT_DIR / f"02_customvoice_ryan_{tag}.wav"
        out.write_bytes(
            cv.synthesize(
                "Same speaker, different style. This demonstrates instruction control.",
                language="English",
                speaker="Ryan",
                instruct=inst,
                do_sample=True,
                temperature=0.85,
                top_p=0.95,
            )
        )
        logger.success(f"Saved: {out.name}")

    # -------------------------------------------------------------------------------------------
    # 2) VoiceDesign (generate new voices from description)
    # -------------------------------------------------------------------------------------------
    logger.info("=== 2) VoiceDesign ===")
    vd = Qwen3TTSAdapter(
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device=DEVICE,
        dtype=DTYPE,
        attn_implementation=ATTN,
    )
    vd.load_model()

    vd_text = "H-hey! You dropped your calculus notebook? I mean, I think it's yours? Maybe?"
    vd_cases = [
        (
            "teen_shy_male",
            "Male, 17 years old, tenor range, shy and nervous; breathy, slight tremble, "
            "gains confidence mid-sentence."
        ),
        (
            "mature_female_anchor",
            "Female, 40s, warm but authoritative news anchor; steady cadence, precise consonants, confident."
        ),
        (
            "cartoon_sidekick",
            "Cartoon sidekick; bright tone, playful, bouncy rhythm, exaggerated vowels and smiles."
        ),
    ]

    voicedesign_refs = {}
    for tag, inst in vd_cases:
        out = OUT_DIR / f"03_voicedesign_{tag}_ref.wav"
        out.write_bytes(
            vd.synthesize(
                vd_text,
                language="English",
                instruct=inst,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
            )
        )
        voicedesign_refs[tag] = out
        logger.success(f"Saved: {out.name}")

    # -------------------------------------------------------------------------------------------
    # 3) Voice Clone (Base): x-vector-only vs ICL, and prompt caching
    # -------------------------------------------------------------------------------------------
    logger.info("=== 3) Voice Clone (Base) ===")
    base = Qwen3TTSAdapter(
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device=DEVICE,
        dtype=DTYPE,
        attn_implementation=ATTN,
    )
    base.load_model()

    # 3.A) x-vector-only (no transcript)
    base.clone_voice(ref_audio=str(REF_AUDIO), x_vector_only_mode=True)
    out = OUT_DIR / "04_clone_xvector_only.wav"
    out.write_bytes(base.synthesize("Hello! This is x-vector-only voice cloning.", language="English"))
    logger.success(f"Saved: {out.name}")

    # Reuse cached prompt without calling clone_voice again (tests the adapter fix)
    out = OUT_DIR / "05_clone_cached_prompt_reuse.wav"
    out.write_bytes(base.synthesize("This line reuses the cached cloning prompt without passing ref_audio.", language="English"))
    logger.success(f"Saved: {out.name}")

    # 3.B) ICL clone (audio + transcript)
    base.clone_voice(ref_audio=str(REF_AUDIO), ref_text=REF_TEXT, x_vector_only_mode=False)
    out = OUT_DIR / "06_clone_icl.wav"
    out.write_bytes(
        base.synthesize(
            "This is ICL cloning using the transcript for improved similarity.",
            language="English",
            do_sample=False,
        )
    )
    logger.success(f"Saved: {out.name}")

    # Decoding sweep (shows controllability via generation kwargs)
    sweep = [
        ("deterministic", dict(do_sample=False)),
        ("mild", dict(do_sample=True, temperature=0.7, top_p=0.9)),
        ("creative", dict(do_sample=True, temperature=0.95, top_p=0.98)),
    ]
    for tag, opts in sweep:
        out = OUT_DIR / f"07_clone_icl_{tag}.wav"
        out.write_bytes(
            base.synthesize(
                "We vary decoding parameters to explore expressiveness and diversity.",
                language="English",
                **opts,
            )
        )
        logger.success(f"Saved: {out.name}")

    # 3.C) VoiceDesign → Clone pipeline: clone a designed voice (using the exact text used to generate it)
    design_ref = voicedesign_refs["mature_female_anchor"]
    base.clone_voice(ref_audio=str(design_ref), ref_text=vd_text, x_vector_only_mode=False)
    out = OUT_DIR / "08_clone_from_voicedesign_ref.wav"
    out.write_bytes(
        base.synthesize(
            "Now we keep the designed timbre, but say completely new content.",
            language="English",
            do_sample=True,
            temperature=0.85,
            top_p=0.95,
        )
    )
    logger.success(f"Saved: {out.name}")

    logger.success("Done.")
