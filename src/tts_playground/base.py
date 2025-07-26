import os
import io
import numpy as np
import soundfile as sf
from abc import ABC, abstractmethod

class BaseTTS(ABC):
    """Abstract base class defining a common interface for TTS models."""
    def __init__(self):
        self.model = None
        self.vocoder = None
        self.sr = None  # sample rate
    
    @abstractmethod
    def load_model(self):
        """Load the TTS model and any necessary components into memory."""
        raise NotImplementedError
    
    @abstractmethod
    def synthesize(self, 
                   text: str,
                   **kwargs) -> bytes:
        """
        Return an audio speech using the voice selected from clone voice
        """
        raise NotImplementedError
    
    def clone_voice(self, voice_sample: str):
        """
        Select a voice or copy one.
        """
        if not os.path.isfile(voice_sample):
            raise FileNotFoundError(f"Voice sample not found: {voice_sample}")
        self._cached_ref = voice_sample
        return True
    
    def _wav_to_bytes(self, waveform: np.ndarray, sample_rate: int) -> bytes:
        """Encode a NumPy float32 (–1..1) or int16 array as 16-bit PCM WAV bytes."""
        buf = io.BytesIO()
        sf.write(buf, waveform, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()