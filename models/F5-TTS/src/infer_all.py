import os
from loguru import logger
import codecs
import argparse

from f5_tts.infer.utils_infer import load_vocoder

from .utils.loader import prepare_model
from .utils.config_loader import load_configs
from .utils.inference import run_inference


def main(config_base_path: str, text: str):
    config = load_configs(config_base_path, config_base_path)
    config.gen_text = text
    config.output_dir = "data/gen"

    path_ref = os.path.dirname(config.voices.main.ref_audio)
    wav_files = [f for f in os.listdir(path_ref) if f.endswith('.wav') and os.path.isfile(os.path.join(path_ref, f))]

    vocoder = load_vocoder(vocoder_name=config.vocoder_name,
                           is_local=config.vocoder_is_local,
                           local_path=config.vocoder_local_path,
                           hf_cache_dir=config.hf_cache_dir)
    logger.info(f"Vocoder '{config.vocoder_name}' loaded ")

    ema_model = prepare_model(model=config.model,
                              model_cfg=config.model_cfg,
                              ckpt_file=config.ckpt_file,
                              vocoder_name=config.vocoder_name,
                              vocab_file=config.vocab_file,
                              cache_dir=config.hf_cache_dir)
    logger.info(f"Model '{config.model}' loaded ")

    for wav_file in wav_files:
        name_file = os.path.splitext(wav_file)[0]
        txt_file = name_file + ".txt"

        config.voices.main.ref_audio = os.path.join(path_ref, wav_file)
        config.voices.main.ref_file = os.path.join(path_ref, txt_file)
        config.voices.main.ref_text = codecs.open(config.voices.main.ref_file, "r", "utf-8").read()
        config.output_file = "infer_" + wav_file

        logger.info(f"Sarting inference of '{name_file}'")

        final_wave, final_sample_rate = run_inference(
            voices_cfg=config.voices,
            gen_text=config.gen_text,
            gen_json=config.gen_json,
            ema_model=ema_model,
            vocoder=vocoder,
            vocoder_name=config.vocoder_name,
            target_rms=config.user_target_rms,
            cross_fade_duration=config.user_cross_fade_duration,
            nfe_step=config.user_nfe_step,
            cfg_strength=config.user_cfg_strength,
            sway_sampling_coef=config.user_sway_sampling_coef,
            speed=config.user_speed,
            fix_duration=config.user_fix_duration,
            save_chunk=config.save_chunk,
            output_dir=config.output_dir,
            output_file=config.output_file,
            remove_silence=config.remove_silence,
        )


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="F5-TTS Inference Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config-base-path',
        type=str,
        default="models/F5-TTS/config/base.yaml",
        help="Path to the base configuration file."
    )

    parser.add_argument(
        '--text',
        type=str,
        default="So, I was, uh, supposed to wake up early today… but then my alarm went off and, uh, I guess I decided that being unconscious was a better plan. Oops.",
        help="Text for generation."
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    logger.debug(f"Received arguments: {args}")

    main(args.config_base_path, args.text)
