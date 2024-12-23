import subprocess
from loguru import logger
from pathlib import Path
import re

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
    return result


def get_audio_mean_volume(input_file, ffmpeg_path='ffmpeg'):
    """
    Analyze the mean volume (in dB) of an audio file using the volumedetect filter.

    Parameters:
        input_file (str or Path): Path to the audio file to analyze.
        ffmpeg_path (str): Path to the ffmpeg executable. Defaults to 'ffmpeg' if in PATH.

    Returns:
        float: The mean volume in dB. If not found, returns None.
    """
    input_file = Path(input_file)

    ffmpeg_cmd = [
        ffmpeg_path,
        '-i', str(input_file),
        '-af', 'volumedetect',
        '-f', 'null',
        '-'
    ]

    result = run_ffmpeg_command(ffmpeg_cmd)

    mean_volume_pattern = re.compile(r"mean_volume:\s+(-?\d+(?:\.\d+)?) dB", re.MULTILINE)

    match = mean_volume_pattern.search(result.stderr)
    mean_volume = None
    if match:
        mean_volume = float(match.group(1))
        logger.info(f"Detected mean volume: {mean_volume} dB in file {input_file}")
    else:
        logger.warning("Could not find mean_volume in the ffmpeg output.")

    return mean_volume


def normalize_audio(input_file, output_file, target_db, ffmpeg_path='ffmpeg'):
    """
    Normalize the audio to a target mean dB by applying a volume filter.

    Parameters:
        input_file (str or Path): Path to the input file.
        output_file (str or Path): Path to the normalized output file.
        target_db (float): The desired mean volume in dB (e.g., -20.0).
        ffmpeg_path (str): Path to the ffmpeg executable. Defaults to 'ffmpeg' if in PATH.
    """
    mean_volume = get_audio_mean_volume(input_file, ffmpeg_path=ffmpeg_path)
    if mean_volume is None:
        logger.warning("No mean volume detected; skipping normalization.")
        return

    # Compute the difference from target_db
    diff_db = target_db - mean_volume
    logger.info(f"Current mean dB: {mean_volume}, target: {target_db}, diff: {diff_db}")

    # If diff_db is close to 0, no real adjustment is needed
    if abs(diff_db) < 0.05:
        logger.info(f"No significant volume change needed for {input_file}")
        # Just copy without changing volume
        ffmpeg_cmd = [
            ffmpeg_path,
            '-y',
            '-i', str(input_file),
            '-c:a', 'copy',
            str(output_file)
        ]
    else:
        # Use the volume filter to apply the difference
        volume_filter = f"volume={diff_db}dB"
        ffmpeg_cmd = [
            ffmpeg_path,
            '-y',
            '-i', str(input_file),
            '-af', volume_filter,
            str(output_file)
        ]

    run_ffmpeg_command(ffmpeg_cmd)
    logger.success(f"Normalized audio saved to {output_file}")