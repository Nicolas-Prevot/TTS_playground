from kokoro import KPipeline
import numpy as np

from tts_core.base import BaseTTS


class KokoroTTSAdapter(BaseTTS):
    def __init__(self,
                 lang_code: str = "a",
                 voice: str = "af_heart"):
        super().__init__()
        self.lang_code = lang_code
        self.voice = voice
        self.pipeline = None
        self.sr = None
        self._cached_embedding = None
    
    def load_model(self):
        self.pipeline = KPipeline(lang_code=self.lang_code)
        self.sr = 24000

    def clone_voice(self, voice: str, lang_code: str | None = None):
        self.voice = voice
        if self.lang_code != lang_code:
            self.lang_code = lang_code
            self.load_model()
        return True
    
    def synthesize(self,
                   text: str,
                   speed=1,
                   split_pattern=r'\n+',) -> bytes:
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        audio_chunks = []
        for _, _, audio in self.pipeline(text, voice=self.voice, speed=speed, split_pattern=split_pattern):
            audio_chunks.append(audio)

        if not audio_chunks:
            raise RuntimeError("No audio produced by Kokoro pipeline.")

        waveform = np.concatenate(audio_chunks, axis=0)
        return self._wav_to_bytes(waveform, self.sr)


if __name__ == "__main__":

    tts = KokoroTTSAdapter(lang_code="a", voice="af_heart")

    tts.load_model()
    tts.clone_voice(lang_code="a", voice="af_bella")

    audio_bytes = tts.synthesize("I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring.",
                                 speed=1.0)

    with open("data/gen/testkokoro.wav", "wb") as f:
        f.write(audio_bytes)
    
    tts.clone_voice(lang_code="f", voice="ff_siwis")

    audio_bytes = tts.synthesize("À l'époque classique, à Athènes, les auteurs doivent présenter au concours trois tragédies et un drame satyrique, les quatre pièces étant jouées par les mêmes acteurs.",
                                 speed=1.0)

    with open("data/gen/testkokorof.wav", "wb") as f:
        f.write(audio_bytes)
