from cached_path import cached_path
from omegaconf import OmegaConf

from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import load_model


def prepare_model(
    model: str, model_cfg: str, ckpt_file: str, vocoder_name: str, vocab_file: str, cache_dir: str
):

    if model == "F5-TTS":
        model_cls = DiT
        model_arch = OmegaConf.load(model_cfg).model.arch
        if not ckpt_file:
            if vocoder_name == "vocos":
                repo_name = "F5-TTS"
                exp_name = "F5TTS_Base"
                ckpt_step = 1200000
                ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors", cache_dir=cache_dir))
            elif vocoder_name == "bigvgan":
                repo_name = "F5-TTS"
                exp_name = "F5TTS_Base_bigvgan"
                ckpt_step = 1250000
                ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.pt", cache_dir=cache_dir))

    elif model == "E2-TTS":
        model_cls = UNetT
        model_arch = dict(dim=1024, depth=24, heads=16, ff_mult=4)
        if not ckpt_file:
            repo_name = "E2-TTS"
            exp_name = "E2TTS_Base"
            ckpt_step = 1200000
            ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors", cache_dir=cache_dir))
    else:
        raise ValueError("Invalid model name")

    ema_model = load_model(model_cls, model_arch, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file)
    return ema_model
