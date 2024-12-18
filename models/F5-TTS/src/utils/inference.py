import os
import re
import numpy as np
import soundfile as sf
from pathlib import Path
from f5_tts.infer.utils_infer import (
    infer_process,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)


def run_inference(
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    config: dict,
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
    remove_silence: bool = False,
    output_dir: str = None,
    output_file: str = None
):
    # Prepare voices dict
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice

    # Preprocess references
    for voice in voices:
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )

    # Split and process gen_text
    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"

    if save_chunk and output_dir and output_file:
        output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
        if not os.path.exists(output_chunk_dir):
            os.makedirs(output_chunk_dir)

    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice_key = match[1]
        else:
            voice_key = "main"

        if voice_key not in voices:
            voice_key = "main"

        text = re.sub(reg2, "", text)
        ref_audio_ = voices[voice_key]["ref_audio"]
        ref_text_ = voices[voice_key]["ref_text"]
        gen_text_ = text.strip()

        audio_segment, final_sample_rate, spectragram = infer_process(
            ref_audio_,
            ref_text_,
            gen_text_,
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

        if save_chunk and output_dir and output_file:
            chunk_text = gen_text_
            if len(chunk_text) > 200:
                chunk_text = chunk_text[:200] + " ... "
            sf.write(
                os.path.join(output_chunk_dir, f"{len(generated_audio_segments)-1}_{chunk_text}.wav"),
                audio_segment,
                final_sample_rate,
            )

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)
    else:
        final_wave = np.array([], dtype=np.float32)
        final_sample_rate = 24000  # default fallback

    # Optionally write to disk
    if output_dir and output_file and len(final_wave) > 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        wave_path = Path(output_dir) / output_file
        sf.write(str(wave_path), final_wave, final_sample_rate)
        if remove_silence:
            remove_silence_for_generated_wav(str(wave_path))
        print(f"Output written to {wave_path}")

    return final_wave, final_sample_rate

