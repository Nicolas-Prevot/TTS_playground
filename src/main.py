import torch, librosa

mars5, config_class = torch.hub.load('Camb-ai/mars5-tts', 'mars5_english', trust_repo=True)
# The `mars5` contains the AR and NAR model, as well as inference code.
# The `config_class` contains tunable inference config settings like temperature.

wav, sr = librosa.load('../data/test1.wav',
                       sr=mars5.sr, mono=True)
wav = torch.from_numpy(wav)
ref_transcript = "Bonjour tout le monde, je m'appelle Nicolas et je suis ici pour vous parler de mon nouveau papier"