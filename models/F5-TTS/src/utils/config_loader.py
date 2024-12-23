import codecs
from importlib.resources import files
from omegaconf import OmegaConf
from pathlib import Path
import json
from loguru import logger


def load_configs(config_base_path, config_path):
    config_base = OmegaConf.load(config_base_path)
    logger.info(f"Base config file loaded from '{config_base_path}'")

    config = OmegaConf.load(config_path)
    logger.info(f"Config file loaded from '{config_path}'")

    conf = OmegaConf.merge(config_base, config)

    if conf.model == "F5-TTS":
        conf.model_cfg = str(files("f5_tts").joinpath("configs/F5TTS_Base_train.yaml"))
    elif conf.model == "E2-TTS":
        conf.model_cfg = str(files("f5_tts").joinpath("configs/F5TTS_Base_train.yaml"))
    else:
        raise ValueError("Invalid model name")

    if conf.gen_file:
        gen_file_path = Path(conf.gen_file)
        if gen_file_path.suffix.lower() == ".json":
            # It's a JSON multi-voice script
            with open(gen_file_path, "r", encoding="utf-8") as f:
                conf.gen_json = json.load(f)
            conf.gen_text = ""
            logger.info(f"Loaded JSON multi-voice script from '{gen_file_path}'")
        else:
            # It's a plain text file
            conf.gen_json = []
            conf.gen_text = codecs.open(conf.gen_file, "r", "utf-8").read()
            logger.info(f"Loaded plain text from '{gen_file_path}'")
    else:
        conf.gen_json = []

    for voice_key, voice_info in conf.voices.items():
        if voice_info.get("ref_file"):
            text_path = Path(voice_info.ref_file)
            voice_text = codecs.open(text_path, "r", "utf-8").read()
            conf.voices[voice_key]["ref_text"] = voice_text

    return conf
