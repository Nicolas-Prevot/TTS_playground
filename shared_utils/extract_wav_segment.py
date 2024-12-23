import argparse
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


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Extract and normalize a segment from an audio file using ffmpeg.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input-file',
        type=Path,
        required=True,
        help="Path to the input audio file (e.g., .wav, .m4a)."
    )

    parser.add_argument(
        '--output-file',
        type=Path,
        default="data/raw_audio/output_cut.wav",
        help="Path to save the extracted audio segment (e.g., output_cut.wav)."
    )

    parser.add_argument(
        '--final-output-file',
        type=Path,
        default="data/raw_audio/output_cutnorm.wav",
        help="Path to save the normalized audio file (e.g., output_cutnorm.wav)."
    )

    parser.add_argument(
        '--start-time',
        type=float,
        default=0.0,
        help="Start time in seconds for the audio segment extraction."
    )

    parser.add_argument(
        '--duration',
        type=float,
        default=30.0,
        help="Duration in seconds of the audio segment to extract."
    )

    parser.add_argument(
        '--target-db',
        type=float,
        default=-20.0,
        help="Target dB level for audio normalization."
    )

    parser.add_argument(
        '--ffmpeg-path',
        type=str,
        default='ffmpeg',
        help="Path to the ffmpeg executable. Defaults to 'ffmpeg' if in PATH."
    )

    return parser.parse_args()


def main(input_wav, extracted_wav, final_wav, start, length, target_db, ffmpeg_path='ffmpeg'):

    extract_audio_segment(input_wav, start, length, extracted_wav, ffmpeg_path=ffmpeg_path)
    normalize_audio(extracted_wav, final_wav, target_db=target_db, ffmpeg_path=ffmpeg_path)


if __name__ == "__main__":

    args = parse_arguments()
    logger.debug(f"Received arguments: {args}")

    main(input_wav=args.input_file,
         extracted_wav=args.output_file,
         final_wav=args.final_output_file,
         start=args.start_time,
         length=args.duration,
         target_db=args.target_db,
         ffmpeg_path=args.ffmpeg_path,
        )

    

    
