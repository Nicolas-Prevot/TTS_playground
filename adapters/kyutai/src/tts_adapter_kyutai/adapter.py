import os
from typing import Optional

import numpy as np
import torch

from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import (
    DEFAULT_DSM_TTS_REPO,
    DEFAULT_DSM_TTS_VOICE_REPO,
    TTSModel,
)

from tts_core.base import BaseTTS


class KyutaiTTSAdapter(BaseTTS):
    """
    Adapter for Kyutai TTS 1.6B (kyutai/tts-1.6b-en_fr) via Moshi.

    Notes:
    - This adapter selects voices by ID from the Kyutai voice bank (kyutai/tts-voices).
    - Audio is decoded during generation using Mimi streaming (on_frame callback),
      which matches Moshi's reference usage pattern.
    """

    def __init__(
        self,
        *,
        hf_repo: str = DEFAULT_DSM_TTS_REPO,
        voice_repo: str = DEFAULT_DSM_TTS_VOICE_REPO,  # kept for clarity / future use
        n_q: int = 32,
        temp: float = 0.6,
        cfg_coef: float = 1.0,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        suppress_torchdynamo_errors: bool = True,
    ):
        super().__init__()

        if suppress_torchdynamo_errors:
            try:
                import torch._dynamo
                torch._dynamo.config.suppress_errors = True
            except Exception:
                pass

        self.hf_repo = hf_repo
        self.voice_repo = voice_repo
        self.n_q = int(n_q)
        self.temp = float(temp)
        self.cfg_coef = float(cfg_coef)

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)

        # Auto dtype: float16 on CUDA, float32 otherwise (can be overridden)
        if dtype is None:
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        else:
            self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        self.model: Optional[TTSModel] = None
        self.sr: Optional[int] = None
        self.selected_voice: Optional[str] = None

    def load_model(self):
        checkpoint_info = CheckpointInfo.from_hf_repo(self.hf_repo)
        self.model = TTSModel.from_checkpoint_info(
            checkpoint_info,
            n_q=self.n_q,
            temp=self.temp,
            device=self.device,
            dtype=self.dtype,
        )
        self.sr = int(self.model.mimi.sample_rate)

    def clone_voice(self, voice_sample: str):
        """
        Kyutai TTS uses a fixed voice bank. `voice_sample` should usually be a voice ID like:
        - vctk/p225_023.wav
        - cml-tts/fr/4724_3731_000031-0001.wav
        - expresso/ex01-ex02_default_001_channel1_168s.wav
        """
        if not isinstance(voice_sample, str) or not voice_sample.strip():
            raise ValueError("voice_sample must be a non-empty string (voice ID or local path).")
        self.selected_voice = voice_sample.strip()
        return True

    def _resolve_voice_path(self, voice_id_or_path: str) -> str:
        assert self.model is not None

        # Allow advanced usage: local wav path
        if os.path.isfile(voice_id_or_path):
            return voice_id_or_path

        # Normal usage: voice ID resolved from kyutai/tts-voices
        try:
            return self.model.get_voice_path(voice_id_or_path)
        except Exception as e:
            # Helpful error message with a sample of available voices
            sample = []
            try:
                sample = self.model.get_voice_names()[:10]
            except Exception:
                pass
            extra = f" Example voices: {sample}" if sample else ""
            raise ValueError(f"Unknown voice id '{voice_id_or_path}'.{extra}") from e

    def synthesize(self, text: str, **kwargs) -> bytes:
        if self.model is None:
            raise RuntimeError("KyutaiTTSAdapter: model not loaded. Call load_model() first.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text input for synthesis must be a non-empty string.")

        # Allow per-call override for cfg_coef + padding
        cfg_coef = float(kwargs.pop("cfg_coef", self.cfg_coef))
        padding_between = int(kwargs.pop("padding_between", 1))

        # If you want to allow more kwargs later, remove this check or explicitly whitelist them.
        if kwargs:
            raise ValueError(f"Unsupported synthesize kwargs: {sorted(kwargs.keys())}")

        entries = self.model.prepare_script([text], padding_between=padding_between)

        if self.selected_voice is None:
            voices = self.model.get_voice_names()
            if not voices:
                raise RuntimeError("No voices found. Is the Kyutai voice repo available in cache?")
            self.selected_voice = voices[0]

        voice_path = self._resolve_voice_path(self.selected_voice)

        cond_attrs = self.model.make_condition_attributes(
            [voice_path],
            cfg_coef=cfg_coef,
        )

        pcm_chunks: list[np.ndarray] = []

        def on_frame(frame: torch.Tensor):
            # Reference behavior: skip invalid frames (-1)
            if (frame != -1).all():
                wav = self.model.mimi.decode(frame[:, 1:, :]).cpu().numpy()[0, 0]
                pcm_chunks.append(np.clip(wav, -1.0, 1.0))

        with torch.inference_mode(), self.model.mimi.streaming(1):
            self.model.generate([entries], [cond_attrs], on_frame=on_frame)

        if not pcm_chunks:
            raise RuntimeError("No audio frames were produced (pcm_chunks is empty).")

        pcm = np.concatenate(pcm_chunks, axis=-1)
        return self._wav_to_bytes(pcm, self.sr or 24000)
