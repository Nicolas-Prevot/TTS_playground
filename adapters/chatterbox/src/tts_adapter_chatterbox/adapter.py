from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from chatterbox.tts import ChatterboxTTS

from tts_core.base import BaseTTS


class ChatterboxTTSAdapter(BaseTTS):
    """Adapter for Resemble AI's Chatterbox TTS (English).

    Design goal (Playground runtime):
    - The worker stages uploaded reference audio into a per-request temp directory.
      That directory is deleted after each request.
    - Upstream Chatterbox can cache voice conditionals in `model.conds` via
      `prepare_conditionals(audio_prompt_path=...)`.
    - We therefore cache conditionals in memory, and synthesize without passing
      `audio_prompt_path`, so we do not depend on the original file path later.
    """

    def __init__(self, device: Optional[str] = None):
        super().__init__()
        self.device = device or self._auto_device()
        self.model: Optional[ChatterboxTTS] = None
        self.sr: Optional[int] = None

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self, *, device: Optional[str] = None) -> None:
        """Load the model weights into memory.

        Args:
            device: Optionally override the device ("cuda", "mps", "cpu").
        """
        if device:
            self.device = device

        # Avoid a common failure mode early.
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        try:
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
        except (AssertionError, RuntimeError) as e:
            # Common failure: CPU-only torch installed but device="cuda"
            msg = str(e).lower()
            if self.device == "cuda" and ("torch not compiled with cuda" in msg or "cuda" in msg):
                self.device = "cpu"
                self.model = ChatterboxTTS.from_pretrained(device="cpu")
            else:
                raise

        self.sr = int(getattr(self.model, "sr", 24000))

    def clone_voice(self, ref_audio: str, *, exaggeration: float = 0.5) -> bool:
        """Prepare and cache voice conditionals from a reference audio clip.

        Caches conditionals in-memory (model.conds) so we don't depend on the path
        existing after the request (temp dir is deleted by the worker).
        """
        if self.model is None:
            raise RuntimeError("Chatterbox model not loaded; call load_model() first.")

        ref_path = Path(ref_audio)
        if not ref_path.is_file():
            raise FileNotFoundError(f"Voice sample not found: {ref_path}")

        self.model.prepare_conditionals(str(ref_path), exaggeration=float(exaggeration))
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

        # Upstream generate() requires existing conditionals if audio_prompt_path is not passed.
        if getattr(self.model, "conds", None) is None:
            raise RuntimeError(
                "No voice conditionals available. Call clone_voice(ref_audio=...) first "
                "or use a pretrained bundle that ships with default conditionals."
            )

        wav_tensor = self.model.generate(
            text,
            repetition_penalty=float(repetition_penalty),
            min_p=float(min_p),
            top_p=float(top_p),
            temperature=float(temperature),
            cfg_weight=float(cfg_weight),
            exaggeration=float(exaggeration),
        )

        wav_np = wav_tensor.squeeze(0).detach().cpu().numpy()
        return self._wav_to_bytes(wav_np, self.sr)
