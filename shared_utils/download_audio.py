from pytubefix import YouTube 
from pytubefix.cli import on_progress


def main(url, save_path):
    yt = YouTube(url, on_progress_callback=on_progress)
    print(yt.title)

    #ys = yt.streams.get_highest_resolution()
    #ys.download(output_path=SAVE_PATH)

    ys = yt.streams.get_audio_only()
    ys.download(output_path=save_path, filename="output.wav")


if __name__ == "__main__":

    save_path = "data/raw_audio"
    url="https://www.youtube.com/watch?v=n434ha4QwU0"  #"https://www.youtube.com/watch?v=F84M0WMY65g"

    main(url, save_path)