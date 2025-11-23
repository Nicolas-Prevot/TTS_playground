import os
import sys
import io
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Callable, Any, Dict
import numpy as np
import soundfile as sf
from indextts.infer_v2 import IndexTTS2

from tts_playground.base import BaseTTS


class IndexTTS2Adapter(BaseTTS):

    def __init__(
        self,
        *,
        # Model bootstrap options (IndexTTS2.__init__)
        model_dir: str = "checkpoints",
        cfg_path: str = "checkpoints/config.yaml",
        use_fp16: bool = False,
        device: Optional[str] = None,
        use_cuda_kernel: Optional[bool] = None,
        use_deepspeed: bool = False,

        # Defaults for inference-time controls (IndexTTS2.infer)
        # You can override any of these per-call via synthesize(...).
        do_sample: bool = True,
        top_p: float = 0.8,
        top_k: Optional[int] = 30,
        temperature: float = 0.8,
        num_beams: int = 3,
        length_penalty: float = 0.0,
        repetition_penalty: float = 10.0,
        max_mel_tokens: int = 1500,

        # Segmentation & audio stitching
        max_text_tokens_per_segment: int = 120,
        interval_silence: int = 200,   # ms of silence between segments

        # Emotion controls (you can also override per-call)
        emo_audio_prompt: Optional[str] = None,   # path to emotion reference audio
        emo_alpha: float = 1.0,                   # blend weight for emotion ref
        emo_vector: Optional[Sequence[float]] = None,  # 8-d vector
        use_emo_text: bool = False,
        emo_text: Optional[str] = None,
        use_random: bool = False,

        # Misc
        verbose: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.cfg_path = cfg_path
        self.use_fp16 = use_fp16
        self.device = device
        self.use_cuda_kernel = use_cuda_kernel
        self.use_deepspeed = use_deepspeed

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

        # non-GPT knobs
        self.max_text_tokens_per_segment = max_text_tokens_per_segment
        self.interval_silence = interval_silence

        # emotion state
        self._emo_audio_prompt = emo_audio_prompt
        self._emo_alpha = emo_alpha
        self._emo_vector = self._validate_emo_vec(emo_vector) if emo_vector is not None else None
        self._use_emo_text = bool(use_emo_text)
        self._emo_text = emo_text
        self._use_random = bool(use_random)

        # misc
        self.verbose = verbose
        self.progress_callback = progress_callback

        # BaseTTS expectations
        self.model: Optional[IndexTTS2] = None
        self.sr: Optional[int] = None  # will be set after first inference (IndexTTS2 uses 22050)
        self._spk_audio_prompt: Optional[str] = None  # set via clone_voice

    def load_model(self):
        """
        Create and initialize IndexTTS2. This loads all submodules and weights.
        """
        self.model = IndexTTS2(
            cfg_path=self.cfg_path,
            model_dir=self.model_dir,
            use_fp16=self.use_fp16,
            device=self.device,
            use_cuda_kernel=self.use_cuda_kernel,
            use_deepspeed=self.use_deepspeed,
        )
        # set optional progress hook for Gradio or CLI progress bars
        if self.progress_callback is not None:
            self.model.gr_progress = self.progress_callback
        # IndexTTS2 vocoder outputs 22050 Hz
        self.sr = 22050

    def clone_voice(self, ref_audio: str):
        """
        Select/set the speaker reference audio ("spk_audio_prompt").
        """
        super().clone_voice(ref_audio)  # validates file path & caches string in BaseTTS
        self._spk_audio_prompt = ref_audio
        # NOTE: IndexTTS2 will refresh its internal caches if the path differs,
        # so we don't force-reset its private caches here.
        return True

    def synthesize(self, text: str, **kwargs) -> bytes:
        """
        Synthesize speech with IndexTTS2.

        Accepts all controls from IndexTTS2.infer() plus GPT generation kwargs.
        Any of these can be passed here to override adapter defaults:

        - emo_audio_prompt: Optional[str]
        - emo_alpha: float
        - emo_vector: Optional[Sequence[float]]  (8 numbers: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm])
        - use_emo_text: bool
        - emo_text: Optional[str]
        - use_random: bool
        - interval_silence: int (ms)
        - verbose: bool
        - max_text_tokens_per_segment: int
        - do_sample, top_p, top_k, temperature, num_beams, length_penalty, repetition_penalty, max_mel_tokens
        - Any extra **generation_kwargs supported by GPT2 generation will be forwarded.
        """
        if self.model is None or self.sr is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self._spk_audio_prompt is None:
            raise ValueError("No speaker voice set. Call clone_voice() first with a reference audio.")

        # merge current defaults with per-call overrides
        gen_kwargs = {**self.gen_kwargs}
        # collect recognized non-GPT kwargs (with adapter state fallbacks)
        emo_audio_prompt = kwargs.pop("emo_audio_prompt", self._emo_audio_prompt)
        emo_alpha = float(kwargs.pop("emo_alpha", self._emo_alpha))
        emo_vector = kwargs.pop("emo_vector", self._emo_vector)
        if emo_vector is not None:
            emo_vector = self._validate_emo_vec(emo_vector)
            # Apply IndexTTS2 recommended bias/normalization:
            emo_vector = self.model.normalize_emo_vec(emo_vector, apply_bias=True)
        use_emo_text = bool(kwargs.pop("use_emo_text", self._use_emo_text))
        emo_text = kwargs.pop("emo_text", self._emo_text)
        use_random = bool(kwargs.pop("use_random", self._use_random))
        interval_silence = int(kwargs.pop("interval_silence", self.interval_silence))
        verbose = bool(kwargs.pop("verbose", self.verbose))
        max_text_tokens_per_segment = int(
            kwargs.pop("max_text_tokens_per_segment", self.max_text_tokens_per_segment)
        )

        # allow per-call generation param overrides
        for k, v in list(kwargs.items()):
            if k in gen_kwargs:
                gen_kwargs[k] = v
                kwargs.pop(k)

        # any remaining kwargs are forwarded directly to IndexTTS2.infer (future-proof)
        forward_kwargs = kwargs

        # Special behavior matches WebUI:
        # If you want "emotion from speaker (same as spk voice)", you must set emo_audio_prompt=None.
        # That is the default in this adapter when _emo_audio_prompt is None.

        # Perform inference into memory (output_path=None returns (sr, np.int16[T, C?]))
        out = self.model.infer(
            spk_audio_prompt=self._spk_audio_prompt,
            text=text,
            output_path=None,
            emo_audio_prompt=emo_audio_prompt,
            emo_alpha=emo_alpha,
            emo_vector=emo_vector,
            use_emo_text=use_emo_text,
            emo_text=emo_text,
            use_random=use_random,
            interval_silence=interval_silence,
            verbose=verbose,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            **gen_kwargs,
            **forward_kwargs,
        )

        if isinstance(out, tuple) and len(out) == 2:
            sr, wav_np = out
            # IndexTTS2 returns int16 PCM as (T, C?) numpy array; BaseTTS accepts int16 or float32
            wav_np = np.asarray(wav_np)
            return self._wav_to_bytes(wav_np, int(sr))
        elif isinstance(out, str) and os.path.isfile(out):
            # very unlikely when output_path=None, but handle just in case
            wav_np, sr = sf.read(out, dtype="int16")
            return self._wav_to_bytes(wav_np, int(sr))
        else:
            # fallback: try to interpret as numpy or torch
            try:
                import torch
                if hasattr(out, "cpu"):
                    wav = out.cpu().numpy()
                else:
                    wav = np.asarray(out)
                return self._wav_to_bytes(wav, self.sr or 22050)
            except Exception as e:
                raise RuntimeError(f"Unexpected infer() return type: {type(out)}") from e

    # ---- Convenience configuration helpers ----------------------------------------------------

    def set_progress_callback(self, cb: Optional[Callable[[float, str], None]]):
        """
        Set a progress callback compatible with IndexTTS2.gr_progress(progress, desc=...).
        """
        self.progress_callback = cb
        if self.model is not None:
            self.model.gr_progress = cb

    def set_generation_options(
        self,
        *,
        do_sample: Optional[bool] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        num_beams: Optional[int] = None,
        length_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_mel_tokens: Optional[int] = None,
        **extra: Any,
    ):
        """
        Update default GPT sampling/beam options used by synthesize().
        Unrecognized kwargs are stored and forwarded to IndexTTS2.infer().
        """
        updates = {
            "do_sample": do_sample,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "max_mel_tokens": max_mel_tokens,
        }
        for k, v in updates.items():
            if v is not None:
                self.gen_kwargs[k] = v
        # keep any forward-compatible extras
        for k, v in extra.items():
            self.gen_kwargs[k] = v

    def set_segmentation(self, *, max_text_tokens_per_segment: Optional[int] = None, interval_silence: Optional[int] = None):
        """
        Configure sentence splitting and inter-segment silence.
        """
        if max_text_tokens_per_segment is not None:
            self.max_text_tokens_per_segment = int(max_text_tokens_per_segment)
        if interval_silence is not None:
            self.interval_silence = int(interval_silence)

    # ---- Emotion configuration helpers --------------------------------------------------------

    def set_emotion_from_audio(self, emo_audio_path: Optional[str], *, emo_alpha: Optional[float] = None):
        """
        Use a separate emotion reference audio; set to None to default to speaker emotion.
        """
        if emo_audio_path is not None and not os.path.isfile(emo_audio_path):
            raise FileNotFoundError(f"Emotion reference audio not found: {emo_audio_path}")
        self._emo_audio_prompt = emo_audio_path
        if emo_alpha is not None:
            self._emo_alpha = float(emo_alpha)
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
        if self.model is None:
            # We'll still store raw; normalize later after load_model()
            norm_vec = list(self._validate_emo_vec(vector8))
        else:
            norm_vec = self.model.normalize_emo_vec(self._validate_emo_vec(vector8), apply_bias=apply_bias)
        self._emo_vector = norm_vec
        if emo_alpha is not None:
            self._emo_alpha = float(emo_alpha)
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
        self._use_emo_text = bool(description)
        self._emo_text = description or None
        if emo_alpha is not None:
            self._emo_alpha = float(emo_alpha)
        # exclusive with other emotion modes
        self._emo_audio_prompt = None
        self._emo_vector = None

    def clear_emotion(self):
        """
        Reset to: 'emotion from speaker voice' (no external reference/vector/text).
        """
        self._emo_audio_prompt = None
        self._emo_alpha = 1.0
        self._emo_vector = None
        self._use_emo_text = False
        self._emo_text = None
        self._use_random = False

    # ---- utilities ----------------------------------------------------------------------------

    @staticmethod
    def _validate_emo_vec(v: Sequence[float]) -> List[float]:
        """
        Ensure the vector has 8 entries and values are floats (clipped to [0, 1.2] like QwenEmotion clamp).
        """
        if v is None:
            return [0.0] * 8
        v = list(v)
        if len(v) != 8:
            raise ValueError(f"Emotion vector must have 8 elements, got {len(v)}.")
        # The QwenEmotion helper clamps to [0.0, 1.2]; do similar pre-clamp here.
        return [float(max(0.0, min(1.2, x))) for x in v]



if __name__ == "__main__":
    tts = IndexTTS2Adapter(
        model_dir="checkpoints/indextts2",
        cfg_path="checkpoints/indextts2/config.yaml",
        use_fp16=False,
        device=None,              # e.g. "cuda:0" or "cpu"; None = auto
        use_cuda_kernel=None,     # None lets IndexTTS2 decide for CUDA; False to force torch ops
        use_deepspeed=False,

        # defaults (can be overridden per call)
        do_sample=True,
        top_p=0.9,
        top_k=30,
        temperature=0.8,
        num_beams=3,
        length_penalty=0.0,
        repetition_penalty=10.0,
        max_mel_tokens=1500,

        max_text_tokens_per_segment=120,
        interval_silence=200,
        verbose=True,
    )

    # Load once
    tts.load_model()

    # Select the speaker reference (your clone voice)
    tts.clone_voice("data/ref/basic_ref_en.wav")

    # Option A: emotion from speaker (default)
    wav_bytes = tts.synthesize("Hello! This is IndexTTS2 via TTS_PLAYGROUND.")
    with open("data/gen/indextts2_demo_speaker.wav", "wb") as f:
        f.write(wav_bytes)

    # Option B: emotion from separate reference audio
    tts.set_emotion_from_audio("data/ref/emo_hate.wav", emo_alpha=0.7)
    wav_bytes = tts.synthesize("Reading with emotions from a different reference voice.")
    with open("data/gen/indextts2_demo_emo_audio.wav", "wb") as f:
        f.write(wav_bytes)

    # Option C: emotion from vector (happy & calm)
    tts.set_emotion_from_vector([0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6], emo_alpha=1.0, use_random=False)
    wav_bytes = tts.synthesize(
        "Custom emotion vectors also work great, with sampling tweaks.",
        temperature=0.7,
        top_p=0.92,
        num_beams=5,
    )
    with open("data/gen/indextts2_demo_emo_vec.wav", "wb") as f:
        f.write(wav_bytes)

    # Option D: emotion from text description
    tts.set_emotion_from_text("gentle, slightly melancholic, intimate", emo_alpha=0.8)
    wav_bytes = tts.synthesize("Il s'agit d'un simple texte en français.") #"A short melancholic passage, softly spoken.")
    with open("data/gen/indextts2_demo_emo_text.wav", "wb") as f:
        f.write(wav_bytes)
