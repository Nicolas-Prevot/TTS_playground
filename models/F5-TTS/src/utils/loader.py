from cached_path import cached_path
from omegaconf import OmegaConf

from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
)


def prepare_vocoder(vocoder_name: str, load_vocoder_from_local: bool):
    if vocoder_name == "vocos":
        vocoder_local_path = "../checkpoints/vocos-mel-24khz"
    elif vocoder_name == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    else:
        raise ValueError("Invalid vocoder name")

    vocoder = load_vocoder(
        vocoder_name=vocoder_name,
        is_local=load_vocoder_from_local,
        local_path=vocoder_local_path,
    )
    return vocoder


def prepare_model(
    model: str, model_cfg: str, ckpt_file: str, vocoder_name: str, vocab_file: str
):
    if model == "F5-TTS":
        model_cls = DiT
        model_arch = OmegaConf.load(model_cfg).model.arch
        if not ckpt_file:  # download from repo
            if vocoder_name == "vocos":
                repo_name = "F5-TTS"
                exp_name = "F5TTS_Base"
                ckpt_step = 1200000
                ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
            elif vocoder_name == "bigvgan":
                repo_name = "F5-TTS"
                exp_name = "F5TTS_Base_bigvgan"
                ckpt_step = 1250000
                ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.pt"))

    elif model == "E2-TTS":
        model_cls = UNetT
        model_arch = dict(dim=1024, depth=24, heads=16, ff_mult=4)
        if not ckpt_file:
            repo_name = "E2-TTS"
            exp_name = "E2TTS_Base"
            ckpt_step = 1200000
            ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
    else:
        raise ValueError("Invalid model name")

    ema_model = load_model(model_cls, model_arch, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file)
    return ema_model
