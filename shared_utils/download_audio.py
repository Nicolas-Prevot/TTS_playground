import os
import argparse
from loguru import logger
from pytubefix import YouTube 
from pytubefix.cli import on_progress


def main(url: str, save_path: str):

    os.makedirs(save_path, exist_ok=True)

    yt = YouTube(url, on_progress_callback=on_progress)
    logger.info(f"Fetching video details for URL: {url}")
    logger.info(f"Video Title: {yt.title}")

    ys = yt.streams.get_audio_only()
    ys.download(output_path=save_path, filename="output.wav")
    logger.success(f"Download completed successfully. File saved at '{os.path.join(save_path, 'output.wav')}'")


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Download audio from a YouTube video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--url',
        type=str,
        required=True,
        help="URL of the YouTube video to download audio from."
    )

    parser.add_argument(
        '--save-path',
        type=str,
        default="data/raw_audio",
        help="Directory where the audio file will be saved."
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    logger.debug(f"Received arguments: {args}")

    main(url=args.url, save_path=args.save_path)
