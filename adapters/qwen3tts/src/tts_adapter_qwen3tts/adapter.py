import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union, List

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel
from tts_core.base import BaseTTS


AudioTuple = Tuple[np.ndarray, int]  # (wav_float32[-1,1], sr)


def _normalize_audio(wav: np.ndarray) -> np.ndarray:
    """
    Normalize audio to float32 in [-1, 1]. Handles int PCM and float.
    Converts multichannel -> mono (mean).
    """
    x = np.asarray(wav)

    if x.ndim > 1:
        x = np.mean(x, axis=-1)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        denom = float(max(abs(info.min), info.max))
        y = x.astype(np.float32) / (denom if denom > 0 else 1.0)
        return np.clip(y, -1.0, 1.0)

    if np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = float(np.max(np.abs(y))) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + 1e-12)
        return np.clip(y, -1.0, 1.0)

    raise TypeError(f"Unsupported audio dtype: {x.dtype}")


def _load_audio_tuple(path: Union[str, Path]) -> AudioTuple:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Reference audio not found: {p}")
    wav, sr = sf.read(str(p), always_2d=False)
    wav = _normalize_audio(wav)
    return wav, int(sr)


def _filter_kwargs_for_callable(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only kwargs accepted by fn (unless fn has **kwargs).
    This makes the adapter resilient to upstream signature drift.
    """
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


def _guess_mode_from_model_id(model_id: str) -> str:
    mid = (model_id or "").lower()
    if "voicedesign" in mid:
        return "voice_design"
    if "customvoice" in mid:
        return "custom_voice"
    # Base == voice clone
    return "voice_clone"


def _parse_dtype(dtype: Optional[Union[str, torch.dtype]], device_map: Optional[str]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        d = dtype.lower().strip()
        if d in ("bf16", "bfloat16"):
            return torch.bfloat16
        if d in ("fp16", "float16", "half"):
            return torch.float16
        if d in ("fp32", "float32"):
            return torch.float32

    # heuristic default
    if device_map and str(device_map).startswith(("cuda", "mps")):
        return torch.bfloat16 if torch.cuda.is_available() else torch.float16
    return torch.float32


@dataclass
class _VoiceCloneState:
    # Upstream naming (preferred): voice_clone_prompt returned by create_voice_clone_prompt(...)
    voice_clone_prompt: Optional[Any] = None
    # Fallback if create_voice_clone_prompt is missing:
    ref_audio: Optional[AudioTuple] = None
    ref_text: Optional[str] = None
    x_vector_only_mode: bool = False


class Qwen3TTSAdapter(BaseTTS):
    """
    TTS Playground adapter for Qwen3-TTS.

    Supports:
    - CustomVoice: generate_custom_voice(text, language, speaker, instruct?, ...)
    - VoiceDesign: generate_voice_design(text, language, instruct, ...)
    - Voice Clone (Base): create_voice_clone_prompt(...) + generate_voice_clone(text, language, voice_clone_prompt=...)
      (or direct ref_audio/ref_text/x_vector_only_mode if prompt caching is unavailable)

    Notes:
    - Upstream uses 'voice_clone_prompt' in generate_voice_clone(...) (per official docs / HF Space).
      Older adapter versions used 'prompt_items', which can be filtered out and cause runtime errors.
      This adapter maps prompt objects to the expected parameter name automatically.
    """

    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        mode: Optional[str] = None,

        # HF / transformers loading knobs (passed to from_pretrained when supported)
        device: Optional[str] = None,              # e.g. "cuda:0", "cuda", "cpu", "mps"
        dtype: Optional[Union[str, torch.dtype]] = None,
        attn_implementation: Optional[str] = "sdpa",  # safe default; upstream suggests flash_attention_2 if installed
        token: Optional[str] = None,

        # Default generation options
        non_streaming_mode: bool = True,
        max_new_tokens: int = 2048,

        # Defaults for CustomVoice / VoiceDesign
        language: str = "Auto",
        speaker: Optional[str] = None,                # for CustomVoice
        instruct: Optional[str] = None,               # for CustomVoice/VoiceDesign

        verbose: bool = False,
    ):
        super().__init__()
        self.model_id = model_id
        self.mode = (mode or _guess_mode_from_model_id(model_id)).strip().lower()

        self.device_map = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _parse_dtype(dtype, self.device_map)
        self.attn_implementation = attn_implementation
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        self.default_language = language
        self.default_speaker = speaker
        self.default_instruct = instruct

        self.gen_defaults: Dict[str, Any] = {
            "non_streaming_mode": bool(non_streaming_mode),
            "max_new_tokens": int(max_new_tokens),
        }

        self.verbose = bool(verbose)

        self.model: Optional[Qwen3TTSModel] = None
        self.sr: Optional[int] = None

        self._vc = _VoiceCloneState()

    # ------------------- lifecycle -------------------

    def load_model(self):
        """
        Load Qwen3-TTS model via Qwen3TTSModel.from_pretrained(...).
        """
        kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "token": self.token,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        filtered = _filter_kwargs_for_callable(Qwen3TTSModel.from_pretrained, kwargs)
        self.model = Qwen3TTSModel.from_pretrained(self.model_id, **filtered)

        self.sr = 24000

    # ------------------- helpers -------------------

    def _resolve_speaker(self, speaker: str) -> str:
        """
        Resolve speaker name robustly across upstream variants (case/underscore differences).
        Prefers exact match from model.get_supported_speakers() when available.
        """
        s = str(speaker).strip()
        if not s:
            return s

        if self.model is None:
            return s

        if hasattr(self.model, "get_supported_speakers"):
            try:
                supported = list(self.model.get_supported_speakers())  # type: ignore[attr-defined]
            except Exception:
                supported = []
            if supported:
                # exact
                if s in supported:
                    return s
                # case-insensitive match
                low_map = {sp.lower(): sp for sp in supported}
                if s.lower() in low_map:
                    return low_map[s.lower()]
                # underscore-normalized match
                s2 = s.replace(" ", "_")
                if s2 in supported:
                    return s2
                if s2.lower() in low_map:
                    return low_map[s2.lower()]

        # fallback normalization: keep case, only spaces->underscore
        return s.replace(" ", "_")

    # ------------------- voice selection / cloning -------------------

    def clone_voice(
        self,
        # Voice Clone (Base)
        ref_audio: Optional[Union[str, Path]] = None,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
        voice_clone_prompt: Optional[Any] = None,

        # CustomVoice / VoiceDesign convenience
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
    ):
        """
        - mode=voice_clone: caches voice_clone_prompt via create_voice_clone_prompt (preferred)
        - mode=custom_voice: caches speaker/language/instruct defaults
        - mode=voice_design: caches language/instruct defaults
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        m = self.mode
        if m == "voice_clone":
            if voice_clone_prompt is not None:
                self._vc = _VoiceCloneState(voice_clone_prompt=voice_clone_prompt)
                return True

            if ref_audio is None:
                raise ValueError("clone_voice(ref_audio=...) is required for mode='voice_clone'.")

            audio_tuple = _load_audio_tuple(ref_audio)

            rt = (ref_text or "").strip()
            if not x_vector_only_mode and not rt:
                raise ValueError("ref_text is required unless x_vector_only_mode=True.")

            if hasattr(self.model, "create_voice_clone_prompt"):
                fn = getattr(self.model, "create_voice_clone_prompt")
                prompt_kwargs: Dict[str, Any] = {
                    "ref_audio": audio_tuple,
                    "ref_text": rt if rt else None,
                    "x_vector_only_mode": bool(x_vector_only_mode),
                }
                prompt_kwargs = _filter_kwargs_for_callable(fn, prompt_kwargs)
                vc_prompt = fn(**prompt_kwargs)

                # store prompt + keep fallback raw ref too (helpful if upstream changes behavior)
                self._vc = _VoiceCloneState(
                    voice_clone_prompt=vc_prompt,
                    ref_audio=audio_tuple,
                    ref_text=rt if rt else None,
                    x_vector_only_mode=bool(x_vector_only_mode),
                )
            else:
                self._vc = _VoiceCloneState(
                    voice_clone_prompt=None,
                    ref_audio=audio_tuple,
                    ref_text=rt if rt else None,
                    x_vector_only_mode=bool(x_vector_only_mode),
                )
            return True

        # custom_voice / voice_design: set defaults
        if language is not None:
            self.default_language = str(language)
        if speaker is not None:
            self.default_speaker = str(speaker)
        if instruct is not None:
            self.default_instruct = str(instruct)

        return True

    # ------------------- synthesis -------------------

    def synthesize(self, text: str, **kwargs) -> bytes:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        m = self.mode

        # merge generation defaults + per-call overrides
        gen_kwargs = dict(self.gen_defaults)
        for k in list(kwargs.keys()):
            if k in gen_kwargs:
                gen_kwargs[k] = kwargs.pop(k)

        language = str(kwargs.pop("language", self.default_language))

        if m == "custom_voice":
            speaker = kwargs.pop("speaker", self.default_speaker)
            if not speaker:
                raise ValueError("CustomVoice requires a speaker. Pass clone_voice(speaker=...) or synthesize(..., speaker=...).")
            instruct = kwargs.pop("instruct", self.default_instruct)

            fn = getattr(self.model, "generate_custom_voice")
            call_kwargs: Dict[str, Any] = {
                "text": text.strip(),
                "language": language,
                "speaker": self._resolve_speaker(speaker),
                "instruct": (str(instruct).strip() if instruct else None),
                **gen_kwargs,
                **kwargs,
            }
            call_kwargs = _filter_kwargs_for_callable(fn, call_kwargs)
            wavs, sr = fn(**call_kwargs)

        elif m == "voice_design":
            instruct = kwargs.pop("instruct", self.default_instruct)
            if not instruct or not str(instruct).strip():
                raise ValueError("VoiceDesign requires an `instruct` voice description.")

            fn = getattr(self.model, "generate_voice_design")
            call_kwargs = {
                "text": text.strip(),
                "language": language,
                "instruct": str(instruct).strip(),
                **gen_kwargs,
                **kwargs,
            }
            call_kwargs = _filter_kwargs_for_callable(fn, call_kwargs)
            wavs, sr = fn(**call_kwargs)

        else:
            # ---------------- voice_clone ----------------
            fn = getattr(self.model, "generate_voice_clone")
            infer_params = inspect.signature(fn).parameters

            # Accept both names from callers; store internally as voice_clone_prompt
            vc_prompt = (
                kwargs.pop("voice_clone_prompt", None)
                or kwargs.pop("prompt_items", None)
                or self._vc.voice_clone_prompt
            )

            # ref audio can be provided per-call, or cached from clone_voice fallback
            ref_audio = kwargs.pop("ref_audio", None)
            ref_text = kwargs.pop("ref_text", None)
            x_vector_only_mode = bool(kwargs.pop("x_vector_only_mode", self._vc.x_vector_only_mode))

            # If caller provides a path, load it
            if isinstance(ref_audio, (str, Path)):
                ref_audio = _load_audio_tuple(ref_audio)

            # If no ref_audio provided, fall back to cached ref_audio (if any)
            if ref_audio is None:
                ref_audio = self._vc.ref_audio

            # If no ref_text provided, fall back to cached ref_text (if any)
            if ref_text is None:
                ref_text = self._vc.ref_text

            # Decide which prompt parameter name upstream accepts
            prompt_key: Optional[str] = None
            if vc_prompt is not None:
                if "voice_clone_prompt" in infer_params:
                    prompt_key = "voice_clone_prompt"
                elif "prompt_items" in infer_params:
                    prompt_key = "prompt_items"
                elif "prompt" in infer_params:
                    prompt_key = "prompt"

            # Validation:
            # Upstream docs: either voice_clone_prompt (preferred) or ref_audio must be provided.
            has_prompt = (prompt_key is not None)
            if not has_prompt and ref_audio is None:
                raise ValueError(
                    "Voice clone requires either a cached voice_clone_prompt (call clone_voice first) "
                    "or ref_audio supplied to synthesize()."
                )
            if not has_prompt and (not x_vector_only_mode) and (not (ref_text and str(ref_text).strip())):
                raise ValueError("ref_text is required unless x_vector_only_mode=True (or use cached voice_clone_prompt).")

            call_kwargs: Dict[str, Any] = {
                "text": text.strip(),
                "language": language,
                "ref_audio": ref_audio,
                "ref_text": (str(ref_text).strip() if ref_text else None),
                "x_vector_only_mode": bool(x_vector_only_mode),
                **gen_kwargs,
                **kwargs,
            }
            if has_prompt and prompt_key is not None:
                call_kwargs[prompt_key] = vc_prompt

            call_kwargs = _filter_kwargs_for_callable(fn, call_kwargs)
            wavs, sr = fn(**call_kwargs)

        # normalize output -> wav bytes
        if not isinstance(wavs, (list, tuple)) or len(wavs) == 0:
            raise RuntimeError(f"Unexpected model output for wavs: {type(wavs)}")

        wav0 = wavs[0]
        if hasattr(wav0, "detach"):
            wav0 = wav0.detach().cpu().numpy()
        wav0 = np.asarray(wav0)

        self.sr = int(sr)
        return self._wav_to_bytes(wav0, self.sr)
