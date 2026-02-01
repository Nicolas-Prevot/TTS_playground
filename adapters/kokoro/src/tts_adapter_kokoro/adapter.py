from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from kokoro import KPipeline

from tts_core.base import BaseTTS


VoiceType = Union[str, "Any"]  # str voice id (e.g. "af_heart") OR a torch tensor


class KokoroTTSAdapter(BaseTTS):
    """
    Kokoro adapter:
    - "clone_voice" selects a predefined voice ID (string) OR optionally loads a local .pt voice tensor.
    - "lang_code" should match the voice (often voice[0]).
    """

    def __init__(self, lang_code: str = "a", voice: str = "af_heart"):
        super().__init__()
        self.lang_code = lang_code
        self.voice: VoiceType = voice

        self.pipeline: Optional[KPipeline] = None
        self.sr = 24000

        # Keep per-language pipelines (cheap) and optionally reuse a shared model (if supported)
        self._pipelines: Dict[str, KPipeline] = {}
        self._shared_model: Any = None

    def _infer_lang_from_voice(self, voice: VoiceType) -> Optional[str]:
        if isinstance(voice, str) and voice:
            return voice[0].lower()
        return None

    def _maybe_load_voice_tensor(self, voice: str) -> VoiceType:
        """
        Optional convenience: if `voice` is a path to a .pt file, load it and store the tensor.
        This helps when you don't want to commit voice files in git.
        """
        p = Path(voice)
        if p.exists() and p.is_file() and p.suffix.lower() == ".pt":
            import torch  # local import to avoid forcing torch import at module import time
            try:
                return torch.load(str(p), weights_only=True)
            except TypeError:
                # older torch versions may not support weights_only
                return torch.load(str(p))
        return voice

    def load_model(self):
        # Idempotent: ensure a pipeline exists for current lang_code
        if self.lang_code in self._pipelines:
            self.pipeline = self._pipelines[self.lang_code]
            return

        # Create a new pipeline; try to reuse the underlying model if the installed kokoro supports it.
        if self._shared_model is None:
            pl = KPipeline(lang_code=self.lang_code)
            self._shared_model = getattr(pl, "model", None)
        else:
            try:
                pl = KPipeline(lang_code=self.lang_code, model=self._shared_model)
            except TypeError:
                # Fallback for older kokoro versions that don't support model=...
                pl = KPipeline(lang_code=self.lang_code)

        self._pipelines[self.lang_code] = pl
        self.pipeline = pl

    def clone_voice(self, voice: str, lang_code: Optional[str] = None):
        # Allow passing a local .pt path for voice embeddings (optional).
        resolved_voice: VoiceType = self._maybe_load_voice_tensor(voice)
        self.voice = resolved_voice

        # If lang_code isn't provided, infer from voice id prefix if possible (e.g. "af_heart" -> "a").
        target_lang = lang_code or self._infer_lang_from_voice(resolved_voice)

        # Only switch language if we have a concrete target (avoid setting None).
        if target_lang and target_lang != self.lang_code:
            self.lang_code = target_lang

        # Ensure pipeline for the (possibly updated) language exists.
        self.load_model()
        return True

    def synthesize(self, text: str, speed: float = 1.0, split_pattern: str = r"\n+") -> bytes:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        audio_chunks = []

        generator = self.pipeline(
            text,
            voice=self.voice,
            speed=speed,
            split_pattern=split_pattern,
        )

        for _, _, audio in generator:
            if audio is None:
                continue

            # audio may be numpy or torch; normalize to numpy float32
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()

            audio = np.asarray(audio, dtype=np.float32)
            if audio.size:
                audio_chunks.append(audio)

        if not audio_chunks:
            raise RuntimeError("No audio produced by Kokoro pipeline.")

        waveform = np.concatenate(audio_chunks, axis=0)
        return self._wav_to_bytes(waveform, self.sr)
