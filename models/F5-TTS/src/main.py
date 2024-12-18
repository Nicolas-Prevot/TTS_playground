from .utils.loader import prepare_vocoder, prepare_model
from .utils.config import load_config, setup_paths_and_defaults
from .utils.inference import run_inference


def main():
    # Load config from file
    config_path = "path/to/your/basic.toml"
    config = load_config(config_path)

    # Set up paths and defaults
    (
        model, model_cfg, ckpt_file, vocab_file, ref_audio, ref_text, gen_text, output_dir, output_file,
        save_chunk, remove_silence, load_vocoder_from_local, vocoder_name, user_target_rms,
        user_cross_fade_duration, user_nfe_step, user_cfg_strength, user_sway_sampling_coef,
        user_speed, user_fix_duration, config
    ) = setup_paths_and_defaults(config)

    # Prepare vocoder and model
    vocoder = prepare_vocoder(vocoder_name, load_vocoder_from_local)
    ema_model = prepare_model(model, model_cfg, ckpt_file, vocoder_name, vocab_file)

    # Run inference programmatically
    final_wave, final_sample_rate = run_inference(
        ref_audio,
        ref_text,
        gen_text,
        config,
        ema_model,
        vocoder,
        vocoder_name,
        target_rms=user_target_rms,
        cross_fade_duration=user_cross_fade_duration,
        nfe_step=user_nfe_step,
        cfg_strength=user_cfg_strength,
        sway_sampling_coef=user_sway_sampling_coef,
        speed=user_speed,
        fix_duration=user_fix_duration,
        save_chunk=save_chunk,
        remove_silence=remove_silence,
        output_dir=output_dir,
        output_file=output_file
    )


if __name__ == "__main__":
    main()