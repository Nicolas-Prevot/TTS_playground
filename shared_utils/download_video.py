from pytubefix import YouTube 
from pytubefix.cli import on_progress

SAVE_PATH = "data/raw_video"
url="https://www.youtube.com/shorts/T96MUBN4gMM"  #"https://www.youtube.com/watch?v=F84M0WMY65g"

yt = YouTube(url, on_progress_callback=on_progress)
print(yt.title)

#ys = yt.streams.get_highest_resolution()
#ys.download(output_path=SAVE_PATH)

ys = yt.streams.get_audio_only()
ys.download(output_path=SAVE_PATH, filename="output.wav")