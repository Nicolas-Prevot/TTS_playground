from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import torch
from loguru import logger

from dia2 import Dia2, GenerationConfig, SamplingConfig  # type: ignore

from tts_core.base import BaseTTS


_UNSET = object()


class Dia2Adapter(BaseTTS):
    """
    Adapter for Nari Labs' Dia2 streaming dialogue TTS model.

    Upstream usage (per official README):
      - Load with: Dia2.from_repo("nari-labs/Dia2-2B", device="cuda", dtype="bfloat16")
      - Generate with: dia.generate(script, config=GenerationConfig(...), ...)

    This adapter wraps that API behind the common BaseTTS interface:
      - load_model(): downloads/loads weights and tokenizer
      - clone_voice(): stores optional prefix speaker WAV paths (conditioning)
      - synthesize(): generates a full WAV (non-streaming) and returns bytes

    Notes
    -----
    * Dia2 expects dialogue scripts with speaker tags, e.g.:
        "[S1] Hello there.\n[S2] Hi!"
    * Prefix conditioning can be supplied for speaker 1 and/or 2.
      Upstream uses Whisper to transcribe prefix files, which adds latency.
    * Upstream keyword is `include_prefix` (older builds may have used
      `include_prefix_audio`). This adapter accepts both.
    """

    def __init__(
        self,
        *,
        repo_id: str = "nari-labs/Dia2-2B",
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        cfg_scale: float = 6.0,
        audio_temperature: float = 0.8,
        audio_top_k: int = 50,
        use_cuda_graph: bool = True,
    ):
        super().__init__()

        self.repo_id = repo_id

        # Match upstream CLI behavior: auto-select CUDA when available.
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # bfloat16 is typically intended for CUDA; on CPU it may fail.
        if self.device != "cuda" and dtype == "bfloat16":
            logger.warning(
                "[Dia2Adapter] dtype=bfloat16 requested on non-CUDA device; "
                "falling back to float32."
            )
            self.dtype = "float32"
        else:
            self.dtype = dtype

        # Defaults
        self._cfg_scale_default = float(cfg_scale)
        self._audio_temp_default = float(audio_temperature)
        self._audio_topk_default = int(audio_top_k)

        # CUDA graph only makes sense on CUDA.
        self._use_cuda_graph_default = bool(use_cuda_graph and self.device == "cuda")

        # Runtime state
        self.model: Optional[Dia2] = None
        self.sr: Optional[int] = None

        # Optional prefix audio prompts for conditioning
        self._prefix_speaker_1: Optional[str] = None
        self._prefix_speaker_2: Optional[str] = None

        # Whether to include the prefix audio at the start of the output.
        self._include_prefix: Optional[bool] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """
        Instantiate Dia2 from the Hugging Face repo id.

        We intentionally do NOT pass a "default_config" to avoid relying on
        internal constructor signatures; instead we build a GenerationConfig
        per call in synthesize().
        """
        logger.info(
            f"[Dia2Adapter] Loading Dia2 from '{self.repo_id}' "
            f"(device={self.device}, dtype={self.dtype})"
        )
        self.model = Dia2.from_repo(self.repo_id, device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Voice conditioning
    # ------------------------------------------------------------------
    def clone_voice(
        self,
        *,
        prefix_speaker_1: Optional[str] = None,
        prefix_speaker_2: Optional[str] = None,
        include_prefix: Optional[bool] = None,
        include_prefix_audio: Optional[bool] = None,
    ) -> bool:
        """
        Store optional WAV paths used for conditioning Dia2 output.

        prefix_speaker_1:
            WAV file path for speaker 1 conditioning.
        prefix_speaker_2:
            WAV file path for speaker 2 conditioning.
        include_prefix / include_prefix_audio:
            If True, the output audio will start with the reference audio clip.
            The upstream keyword is `include_prefix`. `include_prefix_audio` is
            accepted as a compatibility alias and will be ignored if
            `include_prefix` is also provided.
        """
        for p in (prefix_speaker_1, prefix_speaker_2):
            if p is not None and not os.path.isfile(p):
                raise FileNotFoundError(f"Prefix audio not found: {p}")

        if include_prefix is not None and include_prefix_audio is not None:
            if bool(include_prefix) != bool(include_prefix_audio):
                raise ValueError(
                    "clone_voice(): include_prefix and include_prefix_audio disagree; "
                    "provide only one."
                )

        self._prefix_speaker_1 = prefix_speaker_1
        self._prefix_speaker_2 = prefix_speaker_2
        self._include_prefix = include_prefix if include_prefix is not None else include_prefix_audio

        logger.info(
            "[Dia2Adapter] Stored prefix audio: "
            f"s1={prefix_speaker_1}, s2={prefix_speaker_2}, "
            f"include_prefix={self._include_prefix}"
        )
        return True

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        *,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        use_cuda_graph: Optional[bool] = None,
        # Per-call prefix overrides. Use None explicitly to disable cached prefix.
        prefix_speaker_1: Union[Optional[str], object] = _UNSET,
        prefix_speaker_2: Union[Optional[str], object] = _UNSET,
        # Upstream kw is `include_prefix`; we accept both names.
        include_prefix: Union[Optional[bool], object] = _UNSET,
        include_prefix_audio: Union[Optional[bool], object] = _UNSET,
        verbose: bool = False,
        **kwargs: Any,
    ) -> bytes:
        """
        Generate WAV bytes from a Dia2 dialogue script.
        """
        if self.model is None:
            raise RuntimeError("Dia2 model not loaded. Call load_model() first.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("`text` must be a non-empty string.")

        if kwargs:
            logger.warning(f"[Dia2Adapter] Ignoring unsupported kwargs: {sorted(kwargs.keys())}")

        # Resolve effective defaults
        eff_cfg = float(cfg_scale) if cfg_scale is not None else self._cfg_scale_default
        eff_temp = float(temperature) if temperature is not None else self._audio_temp_default
        eff_topk = int(top_k) if top_k is not None else self._audio_topk_default

        eff_cuda_graph = (
            bool(use_cuda_graph) if use_cuda_graph is not None else self._use_cuda_graph_default
        )
        eff_cuda_graph = bool(eff_cuda_graph and self.device == "cuda")

        config = GenerationConfig(
            cfg_scale=eff_cfg,
            audio=SamplingConfig(temperature=eff_temp, top_k=eff_topk),
            use_cuda_graph=eff_cuda_graph,
        )

        # Per-call overrides > cached values.
        if prefix_speaker_1 is _UNSET:
            eff_prefix_1 = self._prefix_speaker_1
        else:
            eff_prefix_1 = prefix_speaker_1  # may be None to disable

        if prefix_speaker_2 is _UNSET:
            eff_prefix_2 = self._prefix_speaker_2
        else:
            eff_prefix_2 = prefix_speaker_2  # may be None to disable

        # include_prefix resolution with aliasing rules:
        # 1) include_prefix (per call)
        # 2) include_prefix_audio (per call, alias)
        # 3) cached clone_voice setting
        if include_prefix is not _UNSET:
            eff_include_prefix = include_prefix
        elif include_prefix_audio is not _UNSET:
            eff_include_prefix = include_prefix_audio
        else:
            eff_include_prefix = self._include_prefix

        logger.info(
            "[Dia2Adapter] Generating "
            f"(cfg_scale={eff_cfg}, temp={eff_temp}, top_k={eff_topk}, "
            f"cuda_graph={eff_cuda_graph}, "
            f"prefix_s1={'set' if eff_prefix_1 else 'none'}, "
            f"prefix_s2={'set' if eff_prefix_2 else 'none'}, "
            f"include_prefix={'yes' if eff_include_prefix else 'no'})"
        )

        gen_kwargs: Dict[str, Any] = {
            "config": config,
            "verbose": verbose,
        }
        if eff_prefix_1 is not None:
            gen_kwargs["prefix_speaker_1"] = eff_prefix_1
        if eff_prefix_2 is not None:
            gen_kwargs["prefix_speaker_2"] = eff_prefix_2
        if eff_include_prefix is not None:
            # Canonical upstream keyword:
            gen_kwargs["include_prefix"] = bool(eff_include_prefix)

        # Call generate, with a compatibility shim for API drift.
        with torch.inference_mode():
            result = self._safe_generate(text, gen_kwargs)

        waveform = getattr(result, "waveform", None)
        sample_rate = int(getattr(result, "sample_rate", 24000))

        if waveform is None:
            raise RuntimeError("Dia2 returned no waveform in GenerationResult.")

        if isinstance(waveform, torch.Tensor):
            wav_np = waveform.detach().cpu().numpy()
        else:
            wav_np = waveform

        # Ensure (T, C) for soundfile
        if getattr(wav_np, "ndim", 0) == 2 and wav_np.shape[0] < wav_np.shape[1]:
            wav_np = wav_np.T

        self.sr = sample_rate
        return self._wav_to_bytes(wav_np, sample_rate)

    def _safe_generate(self, text: str, gen_kwargs: Dict[str, Any]) -> Any:
        """
        Generate with compatibility fallbacks if upstream signature changes.

        Known drift across releases:
          - `include_prefix` vs `include_prefix_audio`
        """
        assert self.model is not None

        def _call(kws: Dict[str, Any]) -> Any:
            return self.model.generate(text, **kws)  # type: ignore[misc]

        try:
            return _call(gen_kwargs)
        except TypeError as e:
            msg = str(e)
            if "unexpected keyword argument" not in msg:
                raise

        # 1) Alias: include_prefix -> include_prefix_audio
        if "include_prefix" in gen_kwargs and "include_prefix_audio" not in gen_kwargs:
            alt = dict(gen_kwargs)
            alt["include_prefix_audio"] = alt.pop("include_prefix")
            try:
                logger.warning(
                    "[Dia2Adapter] Retrying with alias include_prefix_audio "
                    "(upstream expects include_prefix_audio)."
                )
                return _call(alt)
            except TypeError as e2:
                if "unexpected keyword argument" not in str(e2):
                    raise
                gen_kwargs = alt  # continue fallback chain using the aliased dict

        # 2) Drop unsupported kwargs progressively (least important first).
        for k in ("include_prefix_audio", "include_prefix", "prefix_speaker_2", "prefix_speaker_1", "verbose"):
            if k in gen_kwargs:
                logger.warning(
                    f"[Dia2Adapter] Retrying without '{k}' (upstream signature mismatch)."
                )
                alt = dict(gen_kwargs)
                alt.pop(k, None)
                try:
                    return _call(alt)
                except TypeError as e3:
                    if "unexpected keyword argument" in str(e3):
                        gen_kwargs = alt
                        continue
                    raise

        # If we reach here, propagate the last error by retrying once more to get a clean traceback.
        return _call(gen_kwargs)
