import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from loguru import logger

from tts_adapter_qwen3tts.adapter import Qwen3TTSAdapter


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -------------------------------------------------------------------------------------------
# Text pack (translated content). Keep `instruct` in English as requested.
# -------------------------------------------------------------------------------------------

LANGS = [
    # (subdir, adapter_language_string)
    ("en", "English"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("de", "German"),
    ("zh", "Chinese"),
]

TEXT: Dict[str, Dict[str, str]] = {
    "en": {
        "customvoice_base": "Good evening. This is a preset speaker test for Qwen3-TTS CustomVoice.",
        "customvoice_style": "Same speaker, different style. This demonstrates instruction control.",
        "voicedesign_text": "H-hey! You dropped your calculus notebook? I mean, I think it's yours? Maybe?",
        "clone_xvector": "Hello! This is x-vector-only voice cloning.",
        "clone_reuse": "This line reuses the cached cloning prompt without passing ref_audio.",
        "clone_icl": "This is ICL cloning using the transcript for improved similarity.",
        "clone_sweep": "We vary decoding parameters to explore expressiveness and diversity.",
        "clone_from_vd": "Now we keep the designed timbre, but say completely new content.",
    },
    "fr": {
        "customvoice_base": "Bonsoir. Ceci est un test de voix prédéfinie pour Qwen3-TTS CustomVoice.",
        "customvoice_style": "Même voix, style différent. Ceci démontre le contrôle par instruction.",
        "voicedesign_text": "H-hé ! Tu as fait tomber ton cahier de maths ? Enfin… je crois que c’est le tien ? Peut-être ?",
        "clone_xvector": "Bonjour ! Ceci est un clonage vocal en mode x-vector uniquement.",
        "clone_reuse": "Cette phrase réutilise le prompt de clonage mis en cache, sans fournir ref_audio.",
        "clone_icl": "Ceci est un clonage ICL utilisant la transcription pour une meilleure similarité.",
        "clone_sweep": "Nous faisons varier les paramètres de décodage pour explorer l’expressivité et la diversité.",
        "clone_from_vd": "Nous conservons maintenant le timbre conçu, mais nous disons un contenu entièrement nouveau.",
    },
    "es": {
        "customvoice_base": "Buenas noches. Esta es una prueba de voz predefinida para Qwen3-TTS CustomVoice.",
        "customvoice_style": "Misma voz, estilo diferente. Esto demuestra el control por instrucciones.",
        "voicedesign_text": "¿E-eh? ¿Se te cayó tu cuaderno de cálculo? O sea… creo que es tuyo, ¿no? ¿Quizá?",
        "clone_xvector": "¡Hola! Esto es clonación de voz en modo solo x-vector.",
        "clone_reuse": "Esta frase reutiliza el prompt de clonación en caché sin proporcionar ref_audio.",
        "clone_icl": "Esto es clonación ICL usando la transcripción para mejorar la similitud.",
        "clone_sweep": "Variamos los parámetros de decodificación para explorar expresividad y diversidad.",
        "clone_from_vd": "Ahora mantenemos el timbre diseñado, pero decimos contenido totalmente nuevo.",
    },
    "de": {
        "customvoice_base": "Guten Abend. Dies ist ein Test einer voreingestellten Stimme für Qwen3-TTS CustomVoice.",
        "customvoice_style": "Gleiche Stimme, anderer Stil. Das demonstriert die Steuerung per Instruktion.",
        "voicedesign_text": "Ä-äh! Du hast dein Matheheft fallen lassen? Also… ich glaube, das ist deins? Vielleicht?",
        "clone_xvector": "Hallo! Das ist Voice-Cloning im reinen x-vector-Modus.",
        "clone_reuse": "Dieser Satz verwendet den gecachten Cloning-Prompt erneut, ohne ref_audio zu übergeben.",
        "clone_icl": "Das ist ICL-Cloning mit Transkript für höhere Ähnlichkeit.",
        "clone_sweep": "Wir variieren Decoding-Parameter, um Ausdruck und Vielfalt zu erkunden.",
        "clone_from_vd": "Jetzt behalten wir das entworfene Timbre bei, sagen aber völlig neuen Inhalt.",
    },
    "zh": {
        "customvoice_base": "晚上好。这是 Qwen3-TTS CustomVoice 预设音色的测试。",
        "customvoice_style": "同一个音色，不同的风格。这展示了通过指令进行控制。",
        "voicedesign_text": "诶、诶？你的高数笔记本掉了吗？我…我觉得是你的？也许？",
        "clone_xvector": "你好！这是仅使用 x-vector 的声音克隆。",
        "clone_reuse": "这句话会复用缓存的克隆提示词，而不再传入 ref_audio。",
        "clone_icl": "这是 ICL 声音克隆，使用转写文本来提升相似度。",
        "clone_sweep": "我们调整解码参数来探索表达力与多样性。",
        "clone_from_vd": "现在我们保留设计出的音色，但朗读全新的内容。",
    },
}

VOICE_DESIGN_CASES: List[Tuple[str, str]] = [
    (
        "teen_shy_male",
        "Male, 17 years old, tenor range, shy and nervous; breathy, slight tremble, gains confidence mid-sentence.",
    ),
    (
        "mature_female_anchor",
        "Female, 40s, warm but authoritative news anchor; steady cadence, precise consonants, confident.",
    ),
    (
        "cartoon_sidekick",
        "Cartoon sidekick; bright tone, playful, bouncy rhythm, exaggerated vowels and smiles.",
    ),
]

STYLE_SWEEP: List[Tuple[str, str | None]] = [
    ("neutral", None),
    ("excited", "Energetic, enthusiastic, fast pace, lively intonation."),
    ("whisper", "Whispering, intimate, soft, slow and careful."),
]


def _get_customvoice_speakers(cv: Qwen3TTSAdapter) -> List[str]:
    speakers = None
    if hasattr(cv.model, "get_supported_speakers"):
        try:
            speakers = list(cv.model.get_supported_speakers())
        except Exception:
            speakers = None
    if speakers:
        return speakers
    # fallback: official list (may evolve upstream)
    return ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]


def _pick_style_speaker(speakers: List[str]) -> str:
    # Prefer Ryan if present, else take the first
    for s in speakers:
        if s.lower().replace(" ", "_") == "ryan":
            return s
    return speakers[0] if speakers else "Ryan"


def _ensure_text_pack(lang_code: str) -> Dict[str, str]:
    if lang_code not in TEXT:
        raise KeyError(f"Missing translations for language code: {lang_code}")
    return TEXT[lang_code]


def run_for_language(
    *,
    lang_code: str,
    lang_name: str,
    out_dir: Path,
    cv: Qwen3TTSAdapter,
    vd: Qwen3TTSAdapter,
    base: Qwen3TTSAdapter,
    ref_audio: Path,
    ref_text_en: str,
):
    t = _ensure_text_pack(lang_code)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n==================== {lang_code.upper()} ({lang_name}) ====================")
    logger.info(f"Output Dir: {out_dir}")

    # -------------------- 1) CustomVoice --------------------
    speakers = _get_customvoice_speakers(cv)
    logger.info(f"[{lang_code}] CustomVoice speakers ({len(speakers)}): {speakers}")

    for spk in speakers:
        out = out_dir / f"01_customvoice_{spk}.wav"
        out.write_bytes(
            cv.synthesize(
                t["customvoice_base"],
                language=lang_name,
                speaker=spk,
                instruct="Calm, professional, slightly slower pace.",
            )
        )
        logger.success(f"[{lang_code}] Saved: {out.name}")

    style_spk = _pick_style_speaker(speakers)
    for tag, inst in STYLE_SWEEP:
        out = out_dir / f"02_customvoice_{style_spk}_{tag}.wav"
        out.write_bytes(
            cv.synthesize(
                t["customvoice_style"],
                language=lang_name,
                speaker=style_spk,
                instruct=inst,
                do_sample=True,
                temperature=0.85,
                top_p=0.95,
            )
        )
        logger.success(f"[{lang_code}] Saved: {out.name}")

    # -------------------- 2) VoiceDesign --------------------
    voicedesign_refs: Dict[str, Path] = {}
    for tag, inst in VOICE_DESIGN_CASES:
        out = out_dir / f"03_voicedesign_{tag}_ref.wav"
        out.write_bytes(
            vd.synthesize(
                t["voicedesign_text"],
                language=lang_name,
                instruct=inst,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
            )
        )
        voicedesign_refs[tag] = out
        logger.success(f"[{lang_code}] Saved: {out.name}")

    # -------------------- 3) Voice Clone (Base) --------------------
    # 3.A) x-vector-only (no transcript)
    base.clone_voice(ref_audio=str(ref_audio), x_vector_only_mode=True)
    out = out_dir / "04_clone_xvector_only.wav"
    out.write_bytes(base.synthesize(t["clone_xvector"], language=lang_name))
    logger.success(f"[{lang_code}] Saved: {out.name}")

    # Prompt reuse without clone_voice call (tests adapter correctness)
    out = out_dir / "05_clone_cached_prompt_reuse.wav"
    out.write_bytes(base.synthesize(t["clone_reuse"], language=lang_name))
    logger.success(f"[{lang_code}] Saved: {out.name}")

    # 3.B) ICL clone (audio + transcript)
    # IMPORTANT: ref_text must match the audio. Your ref audio is English, so we always use the EN transcript.
    base.clone_voice(ref_audio=str(ref_audio), ref_text=ref_text_en, x_vector_only_mode=False)
    out = out_dir / "06_clone_icl.wav"
    out.write_bytes(base.synthesize(t["clone_icl"], language=lang_name, do_sample=False))
    logger.success(f"[{lang_code}] Saved: {out.name}")

    # Decoding sweep
    sweep = [
        ("deterministic", dict(do_sample=False)),
        ("mild", dict(do_sample=True, temperature=0.7, top_p=0.9)),
        ("creative", dict(do_sample=True, temperature=0.95, top_p=0.98)),
    ]
    for s_tag, opts in sweep:
        out = out_dir / f"07_clone_icl_{s_tag}.wav"
        out.write_bytes(base.synthesize(t["clone_sweep"], language=lang_name, **opts))
        logger.success(f"[{lang_code}] Saved: {out.name}")

    # 3.C) VoiceDesign → Clone pipeline (language-native)
    # Here the ref audio + ref text must match; we generated VoiceDesign refs in this language, so use t["voicedesign_text"].
    design_ref = voicedesign_refs["mature_female_anchor"]
    base.clone_voice(ref_audio=str(design_ref), ref_text=t["voicedesign_text"], x_vector_only_mode=False)
    out = out_dir / "08_clone_from_voicedesign_ref.wav"
    out.write_bytes(
        base.synthesize(
            t["clone_from_vd"],
            language=lang_name,
            do_sample=True,
            temperature=0.85,
            top_p=0.95,
        )
    )
    logger.success(f"[{lang_code}] Saved: {out.name}")


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent

    ROOT_OUT = REPO_ROOT / "data" / "local_examples" / "qwen3tts"
    ROOT_OUT.mkdir(parents=True, exist_ok=True)

    REF_AUDIO = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    # IMPORTANT: for ICL cloning, REF_TEXT must match REF_AUDIO closely.
    REF_TEXT_EN = "Some call me nature, others call me mother nature."

    if not REF_AUDIO.exists():
        logger.error(f"Missing reference audio: {REF_AUDIO}")
        sys.exit(1)

    DEVICE = _default_device()
    DTYPE = "bfloat16" if DEVICE.startswith("cuda") else ("float16" if DEVICE == "mps" else "float32")
    ATTN = "sdpa"  # flash_attention_2 is optional; not installed by default in this adapter

    logger.info(f"Repo Root: {REPO_ROOT}")
    logger.info(f"Root Output Dir: {ROOT_OUT}")
    logger.info(f"Device={DEVICE} dtype={DTYPE} attn_implementation={ATTN}")

    # Load each model ONCE, then reuse for all languages.
    logger.info("Loading adapters (one-time)...")

    cv = Qwen3TTSAdapter(
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device=DEVICE,
        dtype=DTYPE,
        attn_implementation=ATTN,
    )
    cv.load_model()

    vd = Qwen3TTSAdapter(
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device=DEVICE,
        dtype=DTYPE,
        attn_implementation=ATTN,
    )
    vd.load_model()

    base = Qwen3TTSAdapter(
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device=DEVICE,
        dtype=DTYPE,
        attn_implementation=ATTN,
    )
    base.load_model()

    logger.success("All models loaded.")

    # Run language suite
    for code, lang_name in LANGS:
        out_dir = ROOT_OUT / code
        run_for_language(
            lang_code=code,
            lang_name=lang_name,
            out_dir=out_dir,
            cv=cv,
            vd=vd,
            base=base,
            ref_audio=REF_AUDIO,
            ref_text_en=REF_TEXT_EN,
        )

    logger.success("Done (all languages).")
