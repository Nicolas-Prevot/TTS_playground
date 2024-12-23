from pathlib import Path
from loguru import logger

from .utils_audio import run_ffmpeg_command, normalize_audio


def extract_audio_segment(input_file, start_time, duration, output_file, ffmpeg_path='ffmpeg'):
    """
    Extract a segment from an audio file using ffmpeg.

    Parameters:
        input_file (str or Path): Path to the input .wav file.
        start_time (float): Start time in seconds (e.g., 30.0 for 30 seconds).
        duration (float): Length of the segment in seconds (e.g., 10.0 for 10 seconds).
        output_file (str or Path): Path to the output .wav file.
        ffmpeg_path (str): Path to the ffmpeg executable. Defaults to 'ffmpeg' if in PATH.
    """
    input_file = Path(input_file)
    output_file = Path(output_file)

    if not input_file.is_file():
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    ffmpeg_cmd = [
        ffmpeg_path,
        '-y',
        '-ss', str(start_time),
        '-t', str(duration),
        '-i', str(input_file),
        str(output_file)
    ]

    logger.info(f"Extracting {duration} seconds from {start_time}s of {input_file} into {output_file}")
    run_ffmpeg_command(ffmpeg_cmd)
    logger.success(f"Successfully created {output_file}")


if __name__ == "__main__":

    input_wav = "data/raw_audio/output.wav"
    input_wav = "data/raw_audio/tom.m4a"
    start = 0.0
    length = 20
    target_db = -20
    extracted_wav = "data/raw_audio/output_cut.wav"
    final_wav = "data/raw_audio/output_cutnorm.wav"
    ffmpeg_executable = "ffmpeg"

    extract_audio_segment(input_wav, start, length, extracted_wav, ffmpeg_path=ffmpeg_executable)
    normalize_audio(extracted_wav, final_wav, target_db=target_db, ffmpeg_path=ffmpeg_executable)

    
