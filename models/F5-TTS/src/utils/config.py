import codecs
from importlib.resources import files
import tomli
from datetime import datetime
from f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration
)


def load_config(config_path: str):
    with open(config_path, "rb") as f:
        return tomli.load(f)

def setup_paths_and_defaults(
    config: dict,
    model: str = None,
    model_cfg: str = None,
    ckpt_file: str = None,
    vocab_file: str = None,
    ref_audio: str = None,
    ref_text: str = None,
    gen_text: str = None,
    gen_file: str = None,
    output_dir: str = None,
    output_file: str = None,
    save_chunk: bool = False,
    remove_silence: bool = False,
    load_vocoder_from_local: bool = False,
    vocoder_name: str = None,
    user_target_rms: float = None,
    user_cross_fade_duration: float = None,
    user_nfe_step: int = None,
    user_cfg_strength: float = None,
    user_sway_sampling_coef: float = None,
    user_speed: float = None,
    user_fix_duration: float = None,
):
    # Set defaults from config or fallback
    model = model or config.get("model", "F5-TTS")
    model_cfg = model_cfg or config.get("model_cfg", str(files("f5_tts").joinpath("configs/F5TTS_Base_train.yaml")))
    ckpt_file = ckpt_file or config.get("ckpt_file", "")
    vocab_file = vocab_file or config.get("vocab_file", "")
    ref_audio = ref_audio or config.get("ref_audio", "infer/examples/basic/basic_ref_en.wav")
    ref_text = ref_text or config.get("ref_text", "Some call me nature, others call me mother nature.")
    gen_text = gen_text or config.get("gen_text", "Here we generate something just for test.")
    gen_file = gen_file or config.get("gen_file", "")
    output_dir = output_dir or config.get("output_dir", "tests")
    output_file = output_file or config.get(
        "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
    )
    save_chunk = save_chunk or config.get("save_chunk", False)
    remove_silence = remove_silence or config.get("remove_silence", False)
    load_vocoder_from_local = load_vocoder_from_local or config.get("load_vocoder_from_local", False)
    vocoder_name = vocoder_name or config.get("vocoder_name", mel_spec_type)
    user_target_rms = user_target_rms or config.get("target_rms", target_rms)
    user_cross_fade_duration = user_cross_fade_duration or config.get("cross_fade_duration", cross_fade_duration)
    user_nfe_step = user_nfe_step or config.get("nfe_step", nfe_step)
    user_cfg_strength = user_cfg_strength or config.get("cfg_strength", cfg_strength)
    user_sway_sampling_coef = user_sway_sampling_coef or config.get("sway_sampling_coef", sway_sampling_coef)
    user_speed = user_speed or config.get("speed", speed)
    user_fix_duration = user_fix_duration or config.get("fix_duration", fix_duration)

    # patches for pip pkg user
    if "infer/examples/" in ref_audio:
        ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
    if "infer/examples/" in gen_file:
        gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))
    if "voices" in config:
        for voice in config["voices"]:
            voice_ref_audio = config["voices"][voice]["ref_audio"]
            if "infer/examples/" in voice_ref_audio:
                config["voices"][voice]["ref_audio"] = str(files("f5_tts").joinpath(f"{voice_ref_audio}"))

    # if gen_file provided, override gen_text
    if gen_file:
        gen_text = codecs.open(gen_file, "r", "utf-8").read()

    return (
        model, model_cfg, ckpt_file, vocab_file, ref_audio, ref_text, gen_text, output_dir, output_file,
        save_chunk, remove_silence, load_vocoder_from_local, vocoder_name, user_target_rms,
        user_cross_fade_duration, user_nfe_step, user_cfg_strength, user_sway_sampling_coef,
        user_speed, user_fix_duration, config
    )