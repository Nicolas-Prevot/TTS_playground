import os
import numpy as np
import torch

import torch._dynamo                               
torch._dynamo.config.suppress_errors = True

from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel

from tts_core.base import BaseTTS


class KyutaiTTSAdapter(BaseTTS):
    """
    Adapter for Kyutai's real-time streaming TTS (kyutai/tts-1.6b-en_fr).
    Uses the Moshi toolkit to load weights, prepare text, and generate audio.
    """
    def __init__(
        self,
        *,
        hf_repo: str = DEFAULT_DSM_TTS_REPO,
        voice_repo: str = DEFAULT_DSM_TTS_VOICE_REPO,
        n_q: int = 32,
        temp: float = 0.6,
        cfg_coef: float = 1.0,
        device: str = None,
    ):
        super().__init__()
        self.hf_repo = hf_repo
        self.voice_repo = voice_repo
        self.n_q = n_q
        self.temp = temp
        self.cfg_coef = cfg_coef
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.sr = None
        self.selected_voice = None

    def load_model(self):
        checkpoint_info = CheckpointInfo.from_hf_repo(self.hf_repo)

        self.model = TTSModel.from_checkpoint_info(
            checkpoint_info,
            n_q=self.n_q,
            temp=self.temp,
            device=self.device
        )
        self.sr = self.model.mimi.sample_rate

    def clone_voice(self, voice_sample: str):
        if not isinstance(voice_sample, str):
            raise ValueError("voice_sample must be a file path or a voice ID string.")
        
        if os.path.isfile(voice_sample):
            self.selected_voice = voice_sample
        else:
            self.selected_voice = voice_sample
        return True

    def synthesize(self, text: str, **kwargs) -> bytes:
        if self.model is None:
            raise RuntimeError("KyutaiTTSAdapter: model not loaded. Call load_model() first.")
        if not text:
            raise ValueError("Text input for synthesis is empty.")

        entries = self.model.prepare_script([text], padding_between=1)

        if self.selected_voice is None:
            self.selected_voice = self.model.get_voice_names()[0]
        if os.path.isfile(self.selected_voice):
            voice_path = self.selected_voice
        else:
            voice_path = self.model.get_voice_path(self.selected_voice)

        cond_attrs = self.model.make_condition_attributes(
            [voice_path],
            cfg_coef=self.cfg_coef
        )

        result = self.model.generate([entries], [cond_attrs])

        pcm_chunks = []
        with self.model.mimi.streaming(1), torch.no_grad():
            for frame in result.frames[self.model.delay_steps :]:
                wav = self.model.mimi.decode(frame[:, 1:, :]).cpu().numpy()[0, 0]
                pcm_chunks.append(np.clip(wav, -1.0, 1.0))

        pcm = np.concatenate(pcm_chunks, axis=-1)
        return self._wav_to_bytes(pcm, self.sr)


if __name__ == "__main__":

    tts = KyutaiTTSAdapter(
        hf_repo="kyutai/tts-1.6b-en_fr",
        n_q=32,
        temp=0.6,
        cfg_coef=1.0,
        device="cuda"
    )

    tts.load_model()

    tts.clone_voice("vctk/p225_023.wav") #"expresso/ex01-ex02_default_001_channel1_168s.wav")

    audio_en = tts.synthesize("I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring.")
    
    with open("data/gen/testkyutai_en.wav", "wb") as f:
        f.write(audio_en)


    tts.clone_voice("cml-tts/fr/4724_3731_000031-0001.wav")
    french_text = "À l'époque classique, à Athènes, les auteurs doivent présenter au concours trois tragédies et un drame satyrique, les quatre pièces étant jouées par les mêmes acteurs."
    audio_fr = tts.synthesize(french_text)

    with open("data/gen/testkyutai_fr.wav", "wb") as f:
        f.write(audio_fr)