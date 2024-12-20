from pathlib import Path
import subprocess
from loguru import logger


def run_ffmpeg_command(ffmpeg_cmd):
    """
    Run an ffmpeg command using subprocess and handle errors.

    Parameters:
        ffmpeg_cmd (list[str]): The ffmpeg command and arguments as a list.

    Raises:
        RuntimeError: If the ffmpeg command fails.
    """
    logger.debug(f"Running command: {' '.join(ffmpeg_cmd)}")
    result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        logger.error(f"ffmpeg command failed with error:\n{result.stderr}")
        raise RuntimeError(f"ffmpeg command failed: {result.stderr.strip()}")
    else:
        logger.debug("ffmpeg command completed successfully.")


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

    # Ensure the input file exists
    if not input_file.is_file():
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Prepare ffmpeg command
    ffmpeg_cmd = [
        ffmpeg_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-i', str(input_file),
        #'-acodec', 'copy',  # Copy audio codec to avoid re-encoding (if compatible)
        str(output_file)
    ]

    logger.info(f"Extracting {duration} seconds from {start_time}s of {input_file} into {output_file}")
    run_ffmpeg_command(ffmpeg_cmd)
    logger.success(f"Successfully created {output_file}")


if __name__ == "__main__":
    # Example usage - adjust as needed.
    input_wav = "data/raw_audio/output.wav"
    start = 0.0
    length = 4.1
    output_wav = "data/ref/squeezie.wav"

    # If needed, specify a custom ffmpeg path
    ffmpeg_executable = "ffmpeg"

    try:
        extract_audio_segment(input_wav, start, length, output_wav, ffmpeg_path=ffmpeg_executable)
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
