import sys
import re
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import soundfile as sf

# Add the source root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "src"))

from tts_playground.client.tts_client import TTSClient


def run_case(
    client: TTSClient,
    *,
    name: str,
    text: str,
    init: Dict[str, Any],
    clone_voice: Optional[Dict[str, Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    out_dir: Path,
    timeout: float = 900.0,
):
    print(f"\n=== Running API Case: {name} ===")
    dest = out_dir / f"{name}.wav"

    res = client.synth(
        adapter="qwen3tts",
        init=init,
        load_model={},
        clone_voice=clone_voice or {},
        synthesize={"text": text, "kwargs": kwargs or {}},
        dest_path=str(dest),
        wait=True,
        download=True,
        timeout=timeout,
    )

    if res.get("state") == "SUCCESS":
        size = dest.stat().st_size if dest.exists() else 0
        print(f"[{name}] SUCCESS -> {dest.name} ({size} bytes)")
    else:
        print(f"[{name}] FAILED: {res.get('error')}")

    return dest


def chunk_text_by_sentences(text: str, max_chars: int = 240) -> List[str]:
    # Cheap chunker: split on sentence boundaries and pack into <= max_chars
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    chunks: List[str] = []
    buf = ""
    for s in sents:
        if not buf:
            buf = s
            continue
        if len(buf) + 1 + len(s) <= max_chars:
            buf += " " + s
        else:
            chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return chunks


def stitch_wavs(wav_paths: List[Path], out_path: Path, silence_ms: int = 200):
    wavs = []
    sr = None
    for p in wav_paths:
        w, this_sr = sf.read(str(p), dtype="float32")
        if sr is None:
            sr = int(this_sr)
        elif int(this_sr) != sr:
            raise ValueError(f"Sample rate mismatch: {p} is {this_sr}, expected {sr}")
        wavs.append(w)

    if sr is None:
        raise ValueError("No wavs to stitch")

    silence = np.zeros(int(sr * (silence_ms / 1000.0)), dtype=np.float32)
    out = []
    for i, w in enumerate(wavs):
        if i > 0:
            out.append(silence)
        out.append(w)
    out = np.concatenate(out, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out, sr, format="WAV", subtype="PCM_16")


if __name__ == "__main__":
    BASE_URL = "http://localhost:7000"

    OUT_DIR = REPO_ROOT / "data" / "api_examples" / "qwen3tts"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Shared reference audio for Voice Clone tests
    # - If you have a transcript for your ref audio, set SPK_REF_TEXT and use ICL mode.
    # - If not, you can still clone with x_vector_only_mode=True (no transcript needed).
    SPK_REF_PATH = REPO_ROOT / "data" / "ref" / "basic_ref_en.wav"
    SPK_REF_TEXT = "Some call me nature, others call me mother nature."

    if not SPK_REF_PATH.exists():
        print(f"ERROR: Reference file not found: {SPK_REF_PATH}")
        sys.exit(1)

    print(f"Connecting to {BASE_URL}...")
    client = TTSClient(BASE_URL, timeout=300.0)

    try:
        spk_ref_blob = client.pack_file(str(SPK_REF_PATH))

        # -------------------------------
        # 0) Common init blocks
        # -------------------------------
        init_clone = {
            "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "device": "cuda:0",          # or "cpu"
            "dtype": "bfloat16",         # try "float16" if your GPU doesn't like bf16
            #"attn_implementation": "flash_attention_2",  # or "sdpa" / omit
        }

        init_design = {
            "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "device": "cuda:0",
            "dtype": "bfloat16",
            #"attn_implementation": "flash_attention_2",
        }

        init_custom = {
            "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "device": "cuda:0",
            "dtype": "bfloat16",
            #"attn_implementation": "flash_attention_2",
        }

        # =====================================================================================
        # A) Voice Clone tests
        # =====================================================================================

        # 1) x-vector only — no transcript required
        run_case(
            client,
            name="01_clone_xvector_only",
            init=init_clone,
            clone_voice={"ref_audio": spk_ref_blob, "x_vector_only_mode": True},
            text="Hello! This is Qwen3-TTS voice clone in x-vector-only mode.",
            kwargs={"language": "English", "temperature": 0.6, "top_p": 0.9},
            out_dir=OUT_DIR,
        )

        # 2) ICL clone — needs transcript
        run_case(
            client,
            name="02_clone_icl",
            init=init_clone,
            clone_voice={"ref_audio": spk_ref_blob, "ref_text": SPK_REF_TEXT, "x_vector_only_mode": False},
            text="Now we're cloning with an audio+transcript prompt for better timbre and style.",
            kwargs={"language": "English", "temperature": 0.7, "top_p": 0.95},
            out_dir=OUT_DIR,
        )

        # 3) Reuse cached prompt (no clone_voice call) — faster
        run_case(
            client,
            name="03_clone_reuse_cached_prompt",
            init=init_clone,
            clone_voice={},  # IMPORTANT: empty dict => runner does NOT call clone_voice()
            text="This line should reuse the cached voice prompt without re-uploading reference audio.",
            kwargs={"language": "English", "instruct": "Warm, friendly, slightly slower.", "temperature": 0.65, "top_p": 0.9},
            out_dir=OUT_DIR,
        )

        # 4) “Controls sweep” — compare decoding knobs quickly
        sweeps = [
            ("04a_controls_deterministic", {"language": "English", "do_sample": False, "num_beams": 1}),
            ("04b_controls_creative", {"language": "English", "temperature": 0.95, "top_p": 0.98, "do_sample": True}),
            ("04c_controls_constrained", {"language": "English", "temperature": 0.6, "top_p": 0.8, "do_sample": True}),
        ]
        for name, kw in sweeps:
            run_case(
                client,
                name=name,
                init=init_clone,
                clone_voice={},  # reuse cached prompt from step (2)
                text="Same text, different decoding controls. Listen for prosody/stability changes.",
                kwargs=kw,
                out_dir=OUT_DIR,
            )

        # 5) Long text (client-side chunk + stitch), reusing cached voice prompt.
        long_text = (
            "This is a longer voice clone test. We are going to split this paragraph into multiple chunks, "
            "synthesize each chunk separately, then stitch them together into one wav on the client side. "
            "This is useful if you hit model length limits or want more robust outputs. "
            "You can tweak max_chars, temperature, and the inter-chunk silence. "
            "Finally, you can compare this stitched output against a single-shot run."
        )

        chunks = chunk_text_by_sentences(long_text, max_chars=220)
        print(f"\n[LongText] chunks={len(chunks)}")
        chunk_paths = []
        for i, ch in enumerate(chunks, start=1):
            p = run_case(
                client,
                name=f"05_long_chunk_{i:02d}",
                init=init_clone,
                clone_voice={},  # reuse cached prompt
                text=ch,
                kwargs={"language": "English", "temperature": 0.7, "top_p": 0.95},
                out_dir=OUT_DIR,
                timeout=1200.0,
            )
            chunk_paths.append(p)

        stitched = OUT_DIR / "05_long_stitched.wav"
        stitch_wavs(chunk_paths, stitched, silence_ms=250)
        print(f"[LongText] stitched -> {stitched.name}")

        # =====================================================================================
        # B) Voice Design tests
        # =====================================================================================

        # 6) Single VoiceDesign reference (good for design->clone workflows)
        design_ref_text = "H-hey! You dropped your calculus notebook? I mean, I think it's yours? Maybe?"
        design_instruct = (
            "Male, 17 years old, tenor range, gaining confidence; "
            "deeper breath support now, but vowels still tighten when nervous."
        )
        design_path = run_case(
            client,
            name="06_voice_design_reference",
            init=init_design,
            text=design_ref_text,
            kwargs={"language": "English", "instruct": design_instruct, "temperature": 0.75, "top_p": 0.95},
            out_dir=OUT_DIR,
            timeout=1200.0,
        )

        # 7) VoiceDesign: persona sweep
        personas = [
            ("07a_design_warm_narrator", "Female, late 30s, warm documentary narrator, steady pace, crisp diction."),
            ("07b_design_animated_streamer", "Male, early 20s, energetic streamer voice, playful, fast pace, lots of smiles."),
            ("07c_design_noir_detective", "Male, 50s, gravelly noir detective, low pitch, slow and deliberate, tired but sharp."),
        ]
        for name, persona in personas:
            run_case(
                client,
                name=name,
                init=init_design,
                text="This is a voice design sample generated from a persona description.",
                kwargs={"language": "English", "instruct": persona, "temperature": 0.8, "top_p": 0.95},
                out_dir=OUT_DIR,
                timeout=1200.0,
            )

        # 8) Design -> Clone (two-step)
        designed_ref_blob = client.pack_file(str(design_path))
        run_case(
            client,
            name="08_design_then_clone_line1",
            init=init_clone,
            clone_voice={"ref_audio": designed_ref_blob, "ref_text": design_ref_text, "x_vector_only_mode": False},
            text="No problem! I actually kinda finished those already. Want to compare answers?",
            kwargs={"language": "English", "temperature": 0.7, "top_p": 0.95},
            out_dir=OUT_DIR,
        )
        run_case(
            client,
            name="08b_design_then_clone_line2_reuse_cached",
            init=init_clone,
            clone_voice={},  # reuse cached clone prompt
            text="What? No! I mean yes, but not like— I just think you're really precise!",
            kwargs={"language": "English", "instruct": "More flustered, slightly faster, breathier.", "temperature": 0.85, "top_p": 0.95},
            out_dir=OUT_DIR,
        )

        # =====================================================================================
        # C) CustomVoice tests (preset speakers)
        # =====================================================================================

        # 9) Generate one sample per preset speaker
        presets = [
            ("Vivian", "English", "Hi! I'm Vivian. This is a preset voice sample.", "Bright, friendly, smiling tone; medium pace."),
            ("Vivian", "French", "Bonjour, ceci est un test de synthèse vocale avec Qwen3-TTS.", "Bright, friendly, smiling tone; medium pace."),
            ("Serena", "English", "Hello, I'm Serena. Let's keep this calm and reassuring.", "Soft, reassuring, slow and clear."),
            ("Uncle_Fu", "Chinese", "大家好，我是Uncle_Fu。我们开始今天的演示。", "沉稳、亲切、语速稍慢。"),
            ("Dylan", "English", "Yo! Dylan here. Let's keep it energetic.", "High energy, playful, faster pace."),
            ("Eric", "English", "This is Eric. Crisp diction, confident delivery.", "Corporate presenter, confident, medium pace."),
            ("Ryan", "English", "Good evening. This is Ryan, speaking professionally.", "Newsroom delivery, calm, slower pace."),
            ("Aiden", "English", "Hey, I'm Aiden. Casual and friendly.", "Casual, conversational, slightly upbeat."),
            ("Aiden", "French", "Bonjour, ceci est un test de synthèse vocale avec Qwen3-TTS.", "Casual, conversational, slightly upbeat."),
            ("Ono_Anna", "Japanese", "こんにちは。音声合成のデモを始めます。", "明るく丁寧、少しゆっくり。"),
            ("Sohee", "Korean", "안녕하세요. 오늘은 음성 합성 데모를 진행하겠습니다.", "Clear, gentle, customer support vibe."),
        ]

        for i, (speaker, lang, txt, instr) in enumerate(presets, start=1):
            run_case(
                client,
                name=f"09_custom_{i:02d}_{speaker}",
                init=init_custom,
                text=txt,
                kwargs={"language": lang, "speaker": speaker, "instruct": instr, "temperature": 0.7, "top_p": 0.95},
                out_dir=OUT_DIR,
                timeout=1200.0,
            )

        # 10) CustomVoice: language Auto (cross-lingual / adaptive)
        run_case(
            client,
            name="10_customvoice_auto_language",
            init=init_custom,
            text="Bonjour! Then we switch to English. Et maintenant, un peu de français encore.",
            kwargs={"language": "Auto", "speaker": "Vivian", "instruct": "Smooth code-switching; keep the same persona.", "temperature": 0.75, "top_p": 0.95},
            out_dir=OUT_DIR,
            timeout=1200.0,
        )

    finally:
        client.close()
        print("\nDone.")
