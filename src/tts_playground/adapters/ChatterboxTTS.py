import os
import torch
from tts_playground.base import BaseTTS
from chatterbox.tts import ChatterboxTTS

class ChatterboxTTSAdapter(BaseTTS):
    """
    Adapter for Resemble AI's Chatterbox TTS model.
    Supports zero-shot voice cloning via audio prompts and exposes:
      - repetition_penalty
      - min_p, top_p (nucleus sampling)
      - temperature
      - cfg_weight (classifier-free guidance)
      - exaggeration (emotion intensity)
    """

    def __init__(self, device: str = None):
        super().__init__()
        
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        self.model: ChatterboxTTS = None
        self.sr: int = None
        self._cached_audio_prompt: str = None

    def load_model(self):

        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sr = self.model.sr

    def clone_voice(self, ref_audio: str):

        if not os.path.isfile(ref_audio):
            raise FileNotFoundError(f"Voice sample not found: {ref_audio}")
        self._cached_audio_prompt = ref_audio
        return True

    def synthesize(
        self,
        text: str,
        *,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5,
    ) -> bytes:

        if self.model is None or self.sr is None:
            raise RuntimeError("Chatterbox model not loaded; call load_model() first.")

        if self._cached_audio_prompt is None:
            raise RuntimeError("call clone_voice() first")

        wav_tensor = self.model.generate(
            text,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            audio_prompt_path=self._cached_audio_prompt,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )
        wav_np = wav_tensor.squeeze(0).cpu().numpy()
        return self._wav_to_bytes(wav_np, self.sr)

if __name__ == "__main__":

    tts = ChatterboxTTSAdapter()
    tts.load_model()
    tts.clone_voice("data/ref/basic_ref_en.wav")

    audio_bytes = tts.synthesize(
        "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring.",
        #repetition_penalty=1.1,
        #min_p=0.02,
        #top_p=0.9,
        #temperature=1.0,
        cfg_weight=0.5,
        exaggeration=0.5,
    )

    with open("data/gen/output_classic.wav", "wb") as f:
        f.write(audio_bytes)


    tts.clone_voice("data/gen/testkokorof.wav")

    audio_bytes = tts.synthesize(
        "Le trésor de Tourouvre, appelé aussi trésor double de Tourouvre, est un trésor monétaire découvert en 2010 sur le territoire de la commune de Tourouvre, dans le département français de l'Orne, en région Normandie.",
        #repetition_penalty=1.1,
        #min_p=0.02,
        #top_p=0.9,
        #temperature=1.0,
        cfg_weight=0.5,
        exaggeration=0.5,
    )

    with open("data/gen/output_classic_fr.wav", "wb") as f:
        f.write(audio_bytes)