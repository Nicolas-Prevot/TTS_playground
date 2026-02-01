import os
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf
from indextts.infer_v2 import IndexTTS2

from tts_core.base import BaseTTS


class IndexTTS2Adapter(BaseTTS):
    """
    TTS Playground adapter for IndexTTS2 (IndexTeam/index-tts).

    Goals:
    - Match upstream argument names where possible.
    - Be robust across upstream API drift by filtering kwargs via introspection.
    - Provide quality-of-life behavior (emotion mode exclusivity, aliases).
    """

    def __init__(
        self,
        *,
        # Model bootstrap options (IndexTTS2.__init__)
        model_dir: str = "checkpoints/indextts2",
        cfg_path: str = "checkpoints/indextts2/config.yaml",
        use_fp16: bool = False,
        device: Optional[str] = None,
        use_cuda_kernel: Optional[bool] = None,
        use_deepspeed: bool = False,

        # Defaults for inference-time controls (IndexTTS2.infer)
        do_sample: bool = True,
        top_p: float = 0.8,
        top_k: Optional[int] = 30,
        temperature: float = 0.8,
        num_beams: int = 3,
        length_penalty: float = 0.0,
        repetition_penalty: float = 10.0,
        max_mel_tokens: int = 1500,

        # Segmentation & audio stitching
        max_text_tokens_per_sentence: int = 120,
        # Back-compat alias (your previous naming). If provided, it overrides *_per_sentence.
        max_text_tokens_per_segment: Optional[int] = None,
        interval_silence: int = 200,   # ms of silence between sentences

        # Emotion controls (override per-call)
        emo_audio_prompt: Optional[str] = None,
        emo_alpha: float = 1.0,
        emo_vector: Optional[Sequence[float]] = None,
        use_emo_text: bool = False,
        emo_text: Optional[str] = None,
        use_random: bool = False,

        # Misc
        verbose: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        super().__init__()
        self.model_dir = str(model_dir)
        self.cfg_path = str(cfg_path)
        self.use_fp16 = bool(use_fp16)
        self.device = device
        self.use_cuda_kernel = use_cuda_kernel
        self.use_deepspeed = bool(use_deepspeed)

        # generation defaults (merge-able and override-able)
        self.gen_kwargs: Dict[str, Any] = {
            "do_sample": do_sample,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "max_mel_tokens": max_mel_tokens,
        }

        # segmentation defaults
        if max_text_tokens_per_segment is not None:
            max_text_tokens_per_sentence = int(max_text_tokens_per_segment)
        self.max_text_tokens_per_sentence = int(max_text_tokens_per_sentence)
        self.interval_silence = int(interval_silence)

        # emotion state
        self._emo_audio_prompt = emo_audio_prompt
        self._emo_alpha = float(emo_alpha)
        self._emo_vector = self._validate_emo_vec(emo_vector) if emo_vector is not None else None
        self._use_emo_text = bool(use_emo_text)
        self._emo_text = emo_text
        self._use_random = bool(use_random)

        # misc
        self.verbose = bool(verbose)
        self.progress_callback = progress_callback

        # BaseTTS expectations
        self.model: Optional[IndexTTS2] = None
        self.sr: Optional[int] = None  # IndexTTS2 uses 22050
        self._spk_audio_prompt: Optional[str] = None  # set via clone_voice

    # ----------------------------- model lifecycle ---------------------------------------------

    @staticmethod
    def _filter_kwargs_for_callable(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter kwargs for a callable. If callable accepts **kwargs, return all.
        Otherwise return only named parameters.
        """
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        allowed = set(sig.parameters.keys())
        allowed.discard("self")
        return {k: v for k, v in kwargs.items() if k in allowed}

    def load_model(self):
        """
        Create and initialize IndexTTS2. This loads all submodules and weights.
        """
        model_dir = Path(self.model_dir).expanduser().resolve()
        cfg_path = Path(self.cfg_path).expanduser().resolve()

        if not model_dir.exists():
            raise FileNotFoundError(
                f"IndexTTS2 model_dir not found: {model_dir}\n"
                f"Expected checkpoints under: <repo_root>/checkpoints/indextts2/"
            )
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"IndexTTS2 cfg_path not found: {cfg_path}\n"
                f"Expected: <repo_root>/checkpoints/indextts2/config.yaml"
            )

        # Be compatible with upstream naming differences (use_fp16 vs is_fp16, etc.)
        init_kwargs: Dict[str, Any] = {
            "cfg_path": str(cfg_path),
            "model_dir": str(model_dir),
            "device": self.device,
            "use_cuda_kernel": self.use_cuda_kernel,
        }

        ctor_params = inspect.signature(IndexTTS2.__init__).parameters
        # fp16 arg name drift
        if "use_fp16" in ctor_params:
            init_kwargs["use_fp16"] = self.use_fp16
        elif "is_fp16" in ctor_params:
            init_kwargs["is_fp16"] = self.use_fp16

        if "use_deepspeed" in ctor_params:
            init_kwargs["use_deepspeed"] = self.use_deepspeed

        init_kwargs = self._filter_kwargs_for_callable(IndexTTS2.__init__, init_kwargs)
        self.model = IndexTTS2(**init_kwargs)

        # optional progress hook for gradio progress bars
        if self.progress_callback is not None and hasattr(self.model, "gr_progress"):
            self.model.gr_progress = self.progress_callback

        self.sr = 22050

    def clone_voice(self, ref_audio: str):
        """
        Select/set the speaker reference audio (IndexTTS2: spk_audio_prompt).
        """
        super().clone_voice(ref_audio)  # validates file path
        self._spk_audio_prompt = ref_audio
        return True

    # ----------------------------- synthesis ---------------------------------------------------

    @staticmethod
    def _clamp01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    def _normalize_emo_vector(self, vec8: Sequence[float], *, apply_bias: bool = True) -> List[float]:
        """
        Normalize emotion vector the same way upstream does when possible.
        Falls back to a safe normalization (clip + sum normalization) if method is absent.
        """
        vec = self._validate_emo_vec(vec8)
        if self.model is not None and hasattr(self.model, "normalize_emo_vec"):
            try:
                out = self.model.normalize_emo_vec(vec, apply_bias=apply_bias)
            except TypeError:
                out = self.model.normalize_emo_vec(vec)
            return [float(x) for x in list(out)]
        s = sum(vec)
        if s > 1e-8:
            return [float(x / max(1.0, s)) for x in vec]
        return vec

    def synthesize(self, text: str, **kwargs) -> bytes:
        """
        Synthesize speech with IndexTTS2.

        Accepts upstream infer() controls plus generation kwargs.
        Back-compat: also accepts `max_text_tokens_per_segment` (alias of `max_text_tokens_per_sentence`).
        """
        if self.model is None or self.sr is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self._spk_audio_prompt is None:
            raise ValueError("No speaker voice set. Call clone_voice() first with a reference audio.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # --- merge generation defaults with per-call overrides ---
        gen_kwargs = dict(self.gen_kwargs)
        for k in list(kwargs.keys()):
            if k in gen_kwargs:
                gen_kwargs[k] = kwargs.pop(k)

        # --- adapter-level overrides with fallbacks to adapter state ---
        emo_audio_prompt = kwargs.pop("emo_audio_prompt", self._emo_audio_prompt)
        emo_alpha = self._clamp01(float(kwargs.pop("emo_alpha", self._emo_alpha)))
        emo_vector = kwargs.pop("emo_vector", self._emo_vector)
        use_emo_text = bool(kwargs.pop("use_emo_text", self._use_emo_text))
        emo_text = kwargs.pop("emo_text", self._emo_text)
        use_random = bool(kwargs.pop("use_random", self._use_random))

        interval_silence = int(kwargs.pop("interval_silence", self.interval_silence))
        verbose = bool(kwargs.pop("verbose", self.verbose))

        # segmentation: accept both names
        mtts = kwargs.pop("max_text_tokens_per_sentence", None)
        mseg = kwargs.pop("max_text_tokens_per_segment", None)
        if mseg is not None and mtts is None:
            mtts = mseg
        max_text_tokens_per_sentence = int(mtts if mtts is not None else self.max_text_tokens_per_sentence)

        # Emotion-mode exclusivity (matches upstream recommendations):
        # - If using vector OR text emotion, ignore emo_audio_prompt.
        if emo_vector is not None or use_emo_text:
            emo_audio_prompt = None

        # If use_emo_text=True and emo_text is empty, upstream uses `text` itself.
        if use_emo_text and (emo_text is None or not str(emo_text).strip()):
            emo_text = text

        if emo_audio_prompt is not None and not os.path.isfile(str(emo_audio_prompt)):
            raise FileNotFoundError(f"Emotion reference audio not found: {emo_audio_prompt}")

        if emo_vector is not None:
            emo_vector = self._normalize_emo_vector(emo_vector, apply_bias=True)

        infer_kwargs: Dict[str, Any] = {
            "spk_audio_prompt": self._spk_audio_prompt,
            "text": text,
            "output_path": None,
            "emo_audio_prompt": emo_audio_prompt,
            "emo_alpha": emo_alpha,
            "emo_vector": emo_vector,
            "use_emo_text": use_emo_text,
            "emo_text": emo_text,
            "use_random": use_random,
            "interval_silence": interval_silence,
            "verbose": verbose,
        }

        # Upstream name is currently `max_text_tokens_per_sentence` in infer_v2.py.
        infer_params = inspect.signature(self.model.infer).parameters
        if "max_text_tokens_per_sentence" in infer_params:
            infer_kwargs["max_text_tokens_per_sentence"] = max_text_tokens_per_sentence
        elif "max_text_tokens_per_segment" in infer_params:
            infer_kwargs["max_text_tokens_per_segment"] = max_text_tokens_per_sentence

        # Remaining kwargs: forward to infer() if it supports **kwargs.
        forward_kwargs = kwargs

        call_kwargs: Dict[str, Any] = {}
        call_kwargs.update(infer_kwargs)
        call_kwargs.update(gen_kwargs)
        call_kwargs.update(forward_kwargs)

        call_kwargs = self._filter_kwargs_for_callable(self.model.infer, call_kwargs)
        out = self.model.infer(**call_kwargs)

        # --- normalize output to WAV bytes ---
        if isinstance(out, tuple) and len(out) == 2:
            sr, wav_np = out
            wav_np = np.asarray(wav_np)
            return self._wav_to_bytes(wav_np, int(sr))

        if isinstance(out, str) and os.path.isfile(out):
            wav_np, sr = sf.read(out, dtype="int16")
            return self._wav_to_bytes(wav_np, int(sr))

        # fallback: numpy-ish / torch-ish
        try:
            if hasattr(out, "cpu"):
                wav = out.cpu().numpy()
            else:
                wav = np.asarray(out)
            return self._wav_to_bytes(wav, int(self.sr or 22050))
        except Exception as e:
            raise RuntimeError(f"Unexpected infer() return type: {type(out)}") from e

    # ----------------------------- convenience helpers -----------------------------------------

    def set_progress_callback(self, cb: Optional[Callable[[float, str], None]]):
        self.progress_callback = cb
        if self.model is not None and hasattr(self.model, "gr_progress"):
            self.model.gr_progress = cb

    def set_generation_options(self, **updates: Any):
        """Update default generation options forwarded to IndexTTS2.infer(..., **generation_kwargs)."""
        for k, v in updates.items():
            if v is not None:
                self.gen_kwargs[k] = v

    def set_segmentation(
        self,
        *,
        max_text_tokens_per_sentence: Optional[int] = None,
        interval_silence: Optional[int] = None,
    ):
        if max_text_tokens_per_sentence is not None:
            self.max_text_tokens_per_sentence = int(max_text_tokens_per_sentence)
        if interval_silence is not None:
            self.interval_silence = int(interval_silence)

    # ---- Emotion configuration helpers ----

    def set_emotion_from_audio(self, emo_audio_path: Optional[str], *, emo_alpha: Optional[float] = None):
        """
        Use a separate emotion reference audio; set to None to default to speaker emotion.
        """
        if emo_audio_path is not None and not os.path.isfile(emo_audio_path):
            raise FileNotFoundError(f"Emotion reference audio not found: {emo_audio_path}")
        self._emo_audio_prompt = emo_audio_path
        if emo_alpha is not None:
            self._emo_alpha = self._clamp01(float(emo_alpha))
        # disable alternate emotion modes when using audio ref
        self._emo_vector = None
        self._use_emo_text = False
        self._emo_text = None

    def set_emotion_from_vector(
        self,
        vector8: Sequence[float],
        *,
        emo_alpha: Optional[float] = None,
        apply_bias: bool = True,
        use_random: Optional[bool] = None,
    ):
        """
        Use an 8-d emotion vector:
        [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        """
        norm_vec = self._normalize_emo_vector(vector8, apply_bias=apply_bias)
        self._emo_vector = norm_vec
        if emo_alpha is not None:
            self._emo_alpha = self._clamp01(float(emo_alpha))
        if use_random is not None:
            self._use_random = bool(use_random)
        # exclusive with other emotion modes
        self._emo_audio_prompt = None
        self._use_emo_text = False
        self._emo_text = None

    def set_emotion_from_text(self, description: Optional[str], *, emo_alpha: Optional[float] = None):
        """
        Use a natural-language description to guide emotion (QwenEmotion).
        Pass None/"" to disable and return to default behavior.
        """
        self._use_emo_text = bool(description and str(description).strip())
        self._emo_text = description or None
        if emo_alpha is not None:
            self._emo_alpha = self._clamp01(float(emo_alpha))
        # exclusive with other emotion modes
        self._emo_audio_prompt = None
        self._emo_vector = None

    def clear_emotion(self):
        """Reset to: 'emotion from speaker voice' (no external reference/vector/text)."""
        self._emo_audio_prompt = None
        self._emo_alpha = 1.0
        self._emo_vector = None
        self._use_emo_text = False
        self._emo_text = None
        self._use_random = False

    @staticmethod
    def _validate_emo_vec(v: Sequence[float]) -> List[float]:
        """Ensure the vector has 8 entries and values are floats, clipped to [0, 1.2]."""
        if v is None:
            return [0.0] * 8
        v = list(v)
        if len(v) != 8:
            raise ValueError(f"Emotion vector must have 8 elements, got {len(v)}.")
        return [float(max(0.0, min(1.2, x))) for x in v]
