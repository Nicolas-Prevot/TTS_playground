import os
from loguru import logger
import numpy as np
import soundfile as sf
from pathlib import Path
from f5_tts.infer.utils_infer import (
    infer_process,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)


def run_inference(
    voices_cfg: dict,
    gen_text: str,
    gen_json: dict,
    ema_model,
    vocoder,
    vocoder_name: str,
    target_rms: float,
    cross_fade_duration: float,
    nfe_step: int,
    cfg_strength: float,
    sway_sampling_coef: float,
    speed: float,
    fix_duration: float,
    save_chunk: bool = False,
    output_dir: str = None,
    output_file: str = None,
    remove_silence: bool = False,
):
    
    for voice_key, voice_info in voices_cfg.items():
        ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(voice_info.get("ref_audio"), voice_info.get("ref_text"))
        voices_cfg[voice_key]["ref_audio"] = ref_audio_processed
        voices_cfg[voice_key]["ref_text"] = ref_text_processed
    logger.info("All voices have been preprocessed.")

    default_voice_key = list(voices_cfg.keys())[0]

    segments = []
    if gen_json:
        for entry in gen_json:
            voice_key = entry.get("voice")
            text = entry.get("text")
            segments.append((voice_key, text))
    else:
        segments.append((default_voice_key, gen_text))

    chunk_dir = None
    if save_chunk and output_dir and output_file:
        chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
        os.makedirs(chunk_dir, exist_ok=True)

    generated_audio_segments = []
    final_sample_rate = 24000  # default fallback

    for idx, (voice_key, segment_text) in enumerate(segments):

        if voice_key not in voices_cfg:
            logger.warning(f"In segment n°{idx}, voice '{voice_key}' not defined in config.voices. Using '{default_voice_key}' voice instead.")
            voice_key = default_voice_key

        ref_audio_ = voices_cfg[voice_key]["ref_audio"]
        ref_text_ = voices_cfg[voice_key]["ref_text"]

        if not segment_text.strip():
            logger.debug(f"Skipping empty text in segment {idx}.")
            continue

        audio_segment, final_sample_rate, spectragram = infer_process(
            ref_audio_,
            ref_text_,
            segment_text,
            ema_model,
            vocoder,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
        )
        generated_audio_segments.append(audio_segment)

        if chunk_dir:
            chunk_fname = f"{idx:03d}_{voice_key}.wav"
            chunk_out_path = os.path.join(chunk_dir, chunk_fname)
            sf.write(chunk_out_path, audio_segment, final_sample_rate)
            logger.debug(f"Saved chunk {idx} for voice '{voice_key}'")

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)
    else:
        final_wave = np.array([], dtype=np.float32)

    if output_dir and output_file and len(final_wave) > 0:
        os.makedirs(output_dir, exist_ok=True)
        wave_path = Path(output_dir) / output_file
        sf.write(str(wave_path), final_wave, final_sample_rate)
        logger.info(f"Final audio written to {wave_path}")

        if remove_silence:
            remove_silence_for_generated_wav(str(wave_path))
            logger.debug(f"Silence removed from {wave_path}")

    return final_wave, final_sample_rate

