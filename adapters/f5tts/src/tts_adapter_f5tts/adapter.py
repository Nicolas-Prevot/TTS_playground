from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Optional

import torchaudio
from cached_path import cached_path
from hydra.utils import get_class
from loguru import logger
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    device as default_device,
    load_vocoder,
    load_model as f5_load_model,
    preprocess_ref_audio_text,
    infer_batch_process,
    chunk_text,
    mel_spec_type as default_mel_spec_type,
    nfe_step as default_nfe_step,
    cfg_strength as default_cfg_strength,
    sway_sampling_coef as default_sway_sampling_coef,
    speed as default_speed,
    fix_duration as default_fix_duration,
    target_rms as default_target_rms,
    cross_fade_duration as default_cross_fade_duration,
    remove_silence_for_generated_wav,
)

from tts_core.base import BaseTTS


class _NoProgress:
    """Disable tqdm output for f5_tts utils_infer.

    Supports both:
      - progress.tqdm(iterable)  (current upstream)
      - progress(iterable)       (future-proofing)
    """

    def __call__(self, it, *args, **kwargs):
        return it

    def tqdm(self, it, *args, **kwargs):
        return it


@dataclass
class _PreparedRef:
    """Prepared reference stored in-memory."""
    audio: Any  # torch.Tensor [C, T]
    sr: int
    text_clean: str
    text_input: Optional[str]
    audio_hash: str  # md5 of the ORIGINAL ref audio file bytes


class F5TTSAdapter(BaseTTS):
    """
    Adapter for SWivid's F5-TTS / E2-TTS.

    Practical notes
    --------------
    - Cloning works best when you pass both:
        * ref_audio: ~10–20s clean speech
        * ref_text: transcript of that speech
      If ref_text is omitted, upstream preprocessing may run ASR
      (slower + may download extra models).
    - TTS_playground stages uploaded blobs into fresh temp paths per request.
      This adapter hashes the reference audio bytes and caches the prepared
      reference in-memory so repeated requests with the same clip are fast.
    """

    def __init__(
        self,
        *,
        model_name: str = "F5TTS_v1_Base",
        model_cfg_path: Optional[str] = None,
        ckpt_file: str = "",
        vocab_file: str = "",
        vocoder_name: Optional[str] = None,  # "vocos" or "bigvgan"
        load_vocoder_from_local: bool = False,
        vocoder_local_path: str = "",
        device: str = default_device,
        hf_cache_dir: Optional[str] = None,
        # Advanced (forwarded to f5_load_model)
        ode_method: Optional[str] = None,
        use_ema: Optional[bool] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.model_cfg_path = model_cfg_path or str(
            files("f5_tts").joinpath(f"configs/{model_name}.yaml")
        )
        self.ckpt_file = ckpt_file
        self.vocab_file = vocab_file
        self.vocoder_name = vocoder_name  # resolved in load_model() if None
        self.load_vocoder_local = load_vocoder_from_local
        self.vocoder_local_path = vocoder_local_path
        self.device = device
        self.hf_cache_dir = hf_cache_dir
        self.ode_method = ode_method
        self.use_ema = use_ema

        self.model = None
        self.vocoder = None
        self.sr: Optional[int] = None

        self._prepared_ref: Optional[_PreparedRef] = None

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------
    def load_model(self) -> None:
        """Instantiate vocoder + diffusion TTS model."""
        model_cfg = OmegaConf.load(self.model_cfg_path)

        # Resolve mel/vocoder type from config unless explicitly set.
        cfg_mel_spec_type: Optional[str]
        try:
            cfg_mel_spec_type = str(model_cfg.model.mel_spec.mel_spec_type)
        except Exception:
            cfg_mel_spec_type = None

        if self.vocoder_name is None:
            self.vocoder_name = cfg_mel_spec_type or default_mel_spec_type
        elif cfg_mel_spec_type and self.vocoder_name != cfg_mel_spec_type:
            logger.warning(
                f"[F5TTSAdapter] vocoder_name='{self.vocoder_name}' does not match "
                f"model config mel_spec_type='{cfg_mel_spec_type}'. "
                "This may degrade quality or fail unless you also switch to a matching checkpoint variant."
            )

        # Vocoder
        try:
            self.vocoder = load_vocoder(
                vocoder_name=self.vocoder_name,
                is_local=self.load_vocoder_local,
                local_path=self.vocoder_local_path,
                device=self.device,
                hf_cache_dir=self.hf_cache_dir,
            )
        except Exception as e:
            if self.vocoder_name == "bigvgan":
                raise RuntimeError(
                    "[F5TTSAdapter] BigVGAN vocoder requires the official SWivid/F5-TTS "
                    "repository with its BigVGAN submodule available in this environment. "
                    "See adapters/f5tts/README.md (BigVGAN section)."
                ) from e
            raise

        # Model backbone + arch
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arch = model_cfg.model.arch

        # HF checkpoint selection (mirrors upstream patterns)
        repo_name, ckpt_step, ckpt_type = "F5-TTS", 1_250_000, "safetensors"

        if self.model_name == "F5TTS_Base":
            if self.vocoder_name == "vocos":
                ckpt_step = 1_200_000
            elif self.vocoder_name == "bigvgan":
                self.model_name = "F5TTS_Base_bigvgan"
                ckpt_type = "pt"
        elif self.model_name == "E2TTS_Base":
            repo_name = "E2-TTS"
            ckpt_step = 1_200_000

        if not self.ckpt_file:
            hf_uri = f"hf://SWivid/{repo_name}/{self.model_name}/model_{ckpt_step}.{ckpt_type}"
            self.ckpt_file = str(cached_path(hf_uri, cache_dir=self.hf_cache_dir))

        logger.info(f"[F5TTSAdapter] Using checkpoint: {self.ckpt_file}")

        self.model = f5_load_model(
            model_cls,
            model_arch,
            self.ckpt_file,
            mel_spec_type=self.vocoder_name,
            vocab_file=self.vocab_file,
            ode_method=(self.ode_method if self.ode_method is not None else "euler"),
            use_ema=(True if self.use_ema is None else bool(self.use_ema)),
            device=self.device,
        )

        # Upstream sample rate is 24kHz.
        self.sr = 24000

    # ---------------------------------------------------------------------
    # Voice cloning
    # ---------------------------------------------------------------------
    def clone_voice(
        self,
        ref_audio: str,
        ref_text: Optional[str] = None,
        *,
        verbose: bool = False,
    ) -> bool:
        """
        Prepare a reference (audio + transcript) for zero-shot voice cloning.
        """
        if not ref_audio:
            raise ValueError("ref_audio is required.")

        try:
            audio_bytes = Path(ref_audio).read_bytes()
        except Exception as e:
            raise FileNotFoundError(f"ref_audio not found or unreadable: {ref_audio}") from e

        audio_hash = hashlib.md5(audio_bytes).hexdigest()

        if (
            self._prepared_ref is not None
            and self._prepared_ref.audio_hash == audio_hash
            and self._prepared_ref.text_input == ref_text
        ):
            # Same audio bytes (even if staged to a different temp path) + same transcript -> reuse.
            logger.debug("[F5TTSAdapter] Reusing cached prepared reference.")
            return True

        #show_info = print if verbose else (lambda *_a, **_k: None)
        show_info = logger.info if verbose else (lambda *_a, **_k: None)

        # Upstream preprocessing returns a TEMP WAV path (clipped/normalized) + cleaned transcript.
        tmp_wav_path, cleaned_text = preprocess_ref_audio_text(
            ref_audio,
            ref_text,
            show_info=show_info,
        )

        # Load the preprocessed WAV into memory and remove the temp file immediately.
        audio, sr = torchaudio.load(tmp_wav_path)
        try:
            import os
            os.remove(tmp_wav_path)
        except Exception:
            pass

        self._prepared_ref = _PreparedRef(
            audio=audio,
            sr=int(sr),
            text_clean=str(cleaned_text),
            text_input=ref_text,
            audio_hash=audio_hash,
        )
        return True

    # ---------------------------------------------------------------------
    # Synthesis
    # ---------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        *,
        speed: Optional[float] = None,
        nfe_step: Optional[int] = None,
        cfg_strength: Optional[float] = None,
        sway_sampling_coef: Optional[float] = None,
        cross_fade_duration: Optional[float] = None,
        target_rms: Optional[float] = None,
        fix_duration: Optional[float] = None,
        device: Optional[str] = None,
    ) -> bytes:
        """Generate speech for `text` using the prepared reference voice."""
        if self.model is None or self.vocoder is None:
            raise RuntimeError("Model not loaded; call load_model() first.")
        if self._prepared_ref is None:
            raise RuntimeError("No reference prepared; call clone_voice(ref_audio, ref_text) first.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string.")

        params = {
            "speed": default_speed if speed is None else float(speed),
            "nfe_step": default_nfe_step if nfe_step is None else int(nfe_step),
            "cfg_strength": default_cfg_strength if cfg_strength is None else float(cfg_strength),
            "sway_sampling_coef": default_sway_sampling_coef if sway_sampling_coef is None else float(sway_sampling_coef),
            "cross_fade_duration": default_cross_fade_duration if cross_fade_duration is None else float(cross_fade_duration),
            "target_rms": default_target_rms if target_rms is None else float(target_rms),
            "fix_duration": default_fix_duration if fix_duration is None else float(fix_duration),
            "device": self.device if device is None else str(device),
        }

        ref_audio = self._prepared_ref.audio
        ref_sr = self._prepared_ref.sr
        ref_text = self._prepared_ref.text_clean

        ref_dur_s = float(ref_audio.shape[-1]) / float(ref_sr)
        ref_dur_s = max(ref_dur_s, 0.01)

        max_chars = int(len(ref_text.encode("utf-8")) / ref_dur_s * (25.0 - ref_dur_s))
        max_chars = max(50, max_chars)  # safety clamp

        gen_text_batches = chunk_text(text, max_chars=max_chars)

        out = infer_batch_process(
            (ref_audio, ref_sr),
            ref_text,
            gen_text_batches,
            self.model,
            self.vocoder,
            mel_spec_type=self.vocoder_name,
            progress=_NoProgress(),
            **params,
        )
        # f5-tts compatibility:
        # - older versions: infer_batch_process(...) -> (wav_np, sr, spec)
        # - newer versions: infer_batch_process(...) -> generator yielding (wav_np, sr, spec)
        if hasattr(out, "__next__"):
            out = next(out)

        if isinstance(out, tuple) and len(out) == 3:
            wav_np, sr, _spec = out
        elif isinstance(out, tuple) and len(out) == 2:
            wav_np, sr = out
            _spec = None
        else:
            # Fallback: assume out is waveform only
            wav_np = out
            sr = self.sr or 24000
            _spec = None

        self.sr = int(sr)
        return self._wav_to_bytes(wav_np, int(sr))

    def remove_silence_for_generated_wav(self, filename: str) -> None:
        remove_silence_for_generated_wav(filename)
