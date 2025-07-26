import argparse
import importlib
import os
import json
from typing import Any

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "F5-TTS": {
        "module": "adapters.F5_TTS",
        "class": "F5TTSAdapter",
        "init": {"model_name": "F5TTS_v1_Base", "vocoder_name": "vocos"},
        "load": {},
        "clone": {"ref_audio": None, "ref_text": None},
        "synth": {"speed": 1.0},
        "post": {"remove_silence": True},
    },
    "ChatterboxTTS": {
        "module": "adapters.ChatterboxTTS",
        "class": "ChatterboxTTSAdapter",
        "init": {},
        "load": {},
        "clone": {"ref_audio": None},
        "synth": {"cfg_weight": 0.5, "exaggeration": 0.5},
        "post": {},
    },
    "Index-TTS": {
        "module": "adapters.IndexTTS",
        "class": "IndexTTSAdapter",
        "init": {
            "model_dir": "checkpoints/indextts",
            "cfg_path": "checkpoints/indextts/config.yaml",
            "is_fp16": True,
            "device": "cuda",
            "use_cuda_kernel": True,
            "fast": False,
        },
        "load": {},
        "clone": {"ref_audio": None},
        "synth": {
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7,
            "num_beams": 5,
            "fast": False,
        },
        "post": {},
    },
    "Kokoro82M": {
        "module": "adapters.kokoroTTS",
        "class": "KokoroTTSAdapter",
        "init": {},
        "load": {},
        "clone": {"lang_code": "a", "voice": "af_bella"},
        "synth": {"speed": 1.0},
        "post": {},
    },
    "KyutaiTTS": {
        "module": "adapters.KyutaiTTS",
        "class": "KyutaiTTSAdapter",
        "init": {
            "hf_repo": "kyutai/tts-1.6b-en_fr",
            "n_q": 32,
            "temp": 0.6,
            "cfg_coef": 1.0,
            "device": "cuda",
        },
        "load": {},
        "clone": {"voice_sample": "vctk/p225_023.wav"},
        "synth": {},
        "post": {},
    },
    "OpenAudioS1-mini": {
        "module": "adapters.OpenAudioS1MiniTTS",
        "class": "OpenAudioS1MiniAdapter",
        "init": {
            "llama_checkpoint_dir": "checkpoints/openaudio-s1-mini",
            "codec_checkpoint_path": "checkpoints/openaudio-s1-mini/codec.pth",
            "decoder_config_name": "modded_dac_vq",
            "device": "cuda",
            "half": False,
        },
        "load": {},
        "clone": {"ref_audio": None, "ref_text": None},
        "synth": {
            "max_new_tokens": 0,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "temperature": 0.8,
        },
        "post": {},
    },
    "HiggsAudio": {
        "module": "adapters.HiggsAudioTTS",
        "class": "HiggsAudioAdapter",
        "init": {"max_new_tokens":4096},
        "load": {},
        "clone": {
            "ref_audio":None,
            "ref_text":None,
            "scene_prompt":"A clear voice speaking in a quiet room."},
        "synth": {
            "temperature":0.95,
            "top_p":0.9,
            "seed":42
        },
        "post": {},
    },
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a single TTS sample with chosen model")
    p.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys(), help="Which TTS model to use")
    p.add_argument("--ref_audio", required=False, help="Path to reference WAV for cloning")
    p.add_argument("--ref_text", required=False, help="Path to reference text for cloning")
    p.add_argument("--text", required=True, help="Text to synthesize")
    p.add_argument("--out", required=True, help="Output WAV file path")

    # JSON overrides
    p.add_argument("--init_args", default=None, help="JSON dict merged into __init__ kwargs")
    p.add_argument("--load_args", default=None, help="JSON dict merged into load_model kwargs")
    p.add_argument("--clone_args", default=None, help="JSON dict merged into clone_voice kwargs")
    p.add_argument("--synth_args", default=None, help="JSON dict merged into synthesize kwargs")

    return p.parse_args()


def merge_dict(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(base) if base else {}
    if extra:
        out.update(extra)
    return out


def main() -> None:
    args = parse_args()
    spec = MODEL_REGISTRY[args.model]

    def custom_parser(arg_str):
        content=arg_str[2:-2]
        params = content.split(",")
        res = {}
        for param in params:
            [key, value] = param.split(":")
            res[key] = value
        return res

    try:
        j = lambda s: json.loads(s) if s else None
        init_over = j(args.init_args)
        load_over = j(args.load_args)
        clone_over = j(args.clone_args)
        synth_over = j(args.synth_args)
    except:
        j = lambda s: custom_parser(s) if s else None
        init_over = j(args.init_args)
        load_over = j(args.load_args)
        clone_over = j(args.clone_args)
        synth_over = j(args.synth_args)
    
    init_kwargs = merge_dict(spec.get("init", {}), init_over)
    load_kwargs = merge_dict(spec.get("load", {}), load_over)
    clone_kwargs = merge_dict(spec.get("clone", {}), clone_over)
    synth_kwargs = merge_dict(spec.get("synth", {}), synth_over)

    if "ref_audio" in clone_kwargs and args.ref_audio is not None:
        clone_kwargs["ref_audio"] = args.ref_audio
    if "ref_text" in clone_kwargs and args.ref_text is not None:
        if os.path.isfile(args.ref_text):
            with open(args.ref_text, "r", encoding="utf-8") as f:
                clone_kwargs["ref_text"] = f.read()
        else:
            clone_kwargs["ref_text"] = args.ref_text
    
    if os.path.isfile(args.text):
        with open(args.text, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text
    synth_kwargs = merge_dict({"text": text}, synth_kwargs)


    mod = importlib.import_module(spec["module"])
    AdapterClass = getattr(mod, spec["class"])

    tts = AdapterClass(**init_kwargs)
    tts.load_model(**load_kwargs)
    tts.clone_voice(**clone_kwargs)
    audio_bytes = tts.synthesize(**synth_kwargs)

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(audio_bytes)

    if spec.get("post", {}).get("remove_silence") and hasattr(tts, "remove_silence_for_generated_wav"):
        tts.remove_silence_for_generated_wav(args.out)

    print(f"Saved audio to {args.out}")


if __name__ == "__main__":
    main()
