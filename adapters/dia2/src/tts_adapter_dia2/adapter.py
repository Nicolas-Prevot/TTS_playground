from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from dia2 import Dia2, GenerationConfig, SamplingConfig  # type: ignore

from tts_core.base import BaseTTS


class Dia2Adapter(BaseTTS):
    """
    Adapter for Nari Labs' Dia2 (2B) streaming dialogue TTS model.

    This wraps the Dia2 Python API into the common `BaseTTS` interface:

      - `load_model()`  -> load model + default config
      - `clone_voice()` -> store optional audio prompts for speaker conditioning
      - `synthesize()`  -> generate a WAV for a given script

    Notes
    -----
    * Dia2 expects *dialogue scripts* with speaker tags, e.g.:

        "[S1] Hello there.\n[S2] Hi! This is Dia2."

    * Audio prompting / voice conditioning is controlled via
      `prefix_speaker_1` / `prefix_speaker_2` and `include_prefix`
      in the underlying library.
    """

    def __init__(
        self,
        *,
        repo_id: str = "nari-labs/Dia2-2B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        cfg_scale: float = 2.0,
        audio_temperature: float = 0.8,
        audio_top_k: int = 50,
        use_cuda_graph: bool = True,
    ):
        super().__init__()

        # Dia2 config
        self.repo_id = repo_id
        self.device = device
        self.dtype = dtype

        self._cfg_scale_default = cfg_scale
        self._audio_temp_default = audio_temperature
        self._audio_topk_default = audio_top_k
        self._use_cuda_graph_default = use_cuda_graph

        # Will be filled in load_model
        self.model: Optional[Dia2] = None
        self._base_config: Optional[GenerationConfig] = None

        # Optional audio prompts for speaker conditioning
        self._prefix_speaker_1: Optional[str] = None
        self._prefix_speaker_2: Optional[str] = None
        self._include_prefix: Optional[bool] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load_model(self):
        """
        Instantiate the Dia2 engine and a base GenerationConfig.

        The base config uses the recommended settings from the official
        quickstart: cfg_scale ≈ 2.0, audio temperature ≈ 0.8, top_k ≈ 50,
        and CUDA graphs enabled for faster inference. :contentReference[oaicite:1]{index=1}
        """
        logger.info(
            f"[Dia2Adapter] Loading Dia2 from repo '{self.repo_id}' "
            f"on device '{self.device}' (dtype={self.dtype})"
        )

        # Base generation config (text+audio sampling + CFG)
        audio_sampling = SamplingConfig(
            temperature=self._audio_temp_default,
            top_k=self._audio_topk_default,
        )

        self._base_config = GenerationConfig(
            # Text sampling uses its own default (0.6, 50) unless overridden
            audio=audio_sampling,
            cfg_scale=self._cfg_scale_default,
            use_cuda_graph=self._use_cuda_graph_default,
        )

        # Dia2 engine: loads tokenizer, Mimi codec & model weights under the hood
        # (weights are fetched from HF for `repo_id`).
        self.model = Dia2(
            repo=self.repo_id,
            device=self.device,
            dtype=self.dtype,
            default_config=self._base_config,
        )

        # Sample rate will be set on first synth, but we can eager-resolve here
        try:
            self.sr = int(self.model.sample_rate)  # type: ignore[attr-defined]
        except Exception:
            # Fallback until first GenerationResult is produced
            self.sr = None

    # ------------------------------------------------------------------
    # Voice cloning / conditioning
    # ------------------------------------------------------------------
    def clone_voice(
        self,
        *,
        prefix_speaker_1: Optional[str] = None,
        prefix_speaker_2: Optional[str] = None,
        include_prefix_audio: Optional[bool] = None,
    ):
        """
        Store optional audio prompts used for conditioning Dia2's output.

        Parameters
        ----------
        prefix_speaker_1:
            Path to a WAV file representing speaker 1.
        prefix_speaker_2:
            Optional path to a WAV file representing speaker 2.
        include_prefix_audio:
            If True, Dia2 will *include* the prefix audio in the output;
            if False, the prefix is used only as a conditioning signal.
        """
        # NOTE: When called via the Playground API, `prefix_speaker_*`
        # will already be real file paths (RunnerManager stages the blobs).
        self._prefix_speaker_1 = prefix_speaker_1
        self._prefix_speaker_2 = prefix_speaker_2
        self._include_prefix = include_prefix_audio

        logger.info(
            "[Dia2Adapter] Stored prefix audio: "
            f"s1={prefix_speaker_1}, s2={prefix_speaker_2}, "
            f"include_prefix={include_prefix_audio}"
        )
        return True

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        *,
        # High-level knobs (mapped to Dia2 overrides)
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        # Expert overrides (passed directly if provided)
        temp_text: Optional[float] = None,
        topk_text: Optional[int] = None,
        temp_audio: Optional[float] = None,
        topk_audio: Optional[int] = None,
        cfg_filter_k: Optional[int] = None,
        initial_padding: Optional[int] = None,
        use_cuda_graph: Optional[bool] = None,
        # Per-call prefix overrides (optional)
        prefix_speaker_1: Optional[str] = None,
        prefix_speaker_2: Optional[str] = None,
        include_prefix: Optional[bool] = None,
        verbose: bool = False,
        **extra_overrides: Any,
    ) -> bytes:
        """
        Generate a WAV from a Dia2 script.

        Parameters
        ----------
        text:
            Input script. Use tags like `[S1]` and `[S2]` to mark speakers,
            e.g. "[S1] Hello. [S2] Hi!".

        temperature, top_k:
            Convenience parameters applied to AUDIO sampling
            (`temp_audio` / `topk_audio`) if the expert args are not used.

        cfg_scale, cfg_filter_k, initial_padding, use_cuda_graph, temp_* etc:
            Forwarded into Dia2's `GenerationConfig` via the built-in
            override mechanism.

        prefix_speaker_*, include_prefix:
            Per-call overrides for the audio prompts. If omitted, the values
            from `clone_voice()` (if any) are used.
        """
        if self.model is None:
            raise RuntimeError("Dia2 model not loaded. Call load_model() first.")
        if self._base_config is None:
            raise RuntimeError("Base GenerationConfig not initialized.")

        if not text or not isinstance(text, str):
            raise ValueError("`text` must be a non-empty string.")

        # 1) Build override dict for GenerationConfig
        overrides: Dict[str, Any] = {}

        # "Expert" fields first (if provided explicitly)
        if temp_text is not None:
            overrides["temp_text"] = float(temp_text)
        if topk_text is not None:
            overrides["topk_text"] = int(topk_text)
        if temp_audio is not None:
            overrides["temp_audio"] = float(temp_audio)
        if topk_audio is not None:
            overrides["topk_audio"] = int(topk_audio)
        if cfg_scale is not None:
            overrides["cfg_scale"] = float(cfg_scale)
        if cfg_filter_k is not None:
            overrides["cfg_filter_k"] = int(cfg_filter_k)
        if initial_padding is not None:
            overrides["initial_padding"] = int(initial_padding)
        if use_cuda_graph is not None:
            overrides["use_cuda_graph"] = bool(use_cuda_graph)

        # High-level shorthands (apply to audio sampling if expert fields unset)
        if temperature is not None and "temp_audio" not in overrides:
            overrides["temp_audio"] = float(temperature)
        if top_k is not None and "topk_audio" not in overrides:
            overrides["topk_audio"] = int(top_k)

        # Extra keys (future-proofing; Dia2 just ignores unknown override keys)
        overrides.update(extra_overrides)

        # 2) Resolve prefix configuration (per-call overrides > cloned defaults)
        eff_prefix_s1 = prefix_speaker_1 or self._prefix_speaker_1
        eff_prefix_s2 = prefix_speaker_2 or self._prefix_speaker_2
        eff_include_prefix = (
            include_prefix if include_prefix is not None else self._include_prefix
        )

        logger.info(
            "[Dia2Adapter] Generating audio "
            f"(temp_audio={overrides.get('temp_audio')}, "
            f"topk_audio={overrides.get('topk_audio')}, "
            f"cfg_scale={overrides.get('cfg_scale', self._cfg_scale_default)}, "
            f"prefix_s1={eff_prefix_s1}, prefix_s2={eff_prefix_s2}, "
            f"include_prefix={eff_include_prefix})"
        )

        # 3) Call Dia2.generate
        result = self.model.generate(  # type: ignore[call-arg]
            text,
            config=self._base_config,
            prefix_speaker_1=eff_prefix_s1,
            prefix_speaker_2=eff_prefix_s2,
            include_prefix=eff_include_prefix,
            verbose=verbose,
            **overrides,
        )

        # 4) Convert waveform -> bytes
        waveform = result.waveform.detach().cpu().numpy()  # type: ignore[attr-defined]
        sample_rate = int(result.sample_rate)  # type: ignore[attr-defined]
        self.sr = sample_rate

        # Dia2 may return (C, T) or (T, C); make sure we give soundfile (T, C)
        if waveform.ndim == 2:
            # Heuristic: the smaller dimension is almost certainly channels
            if waveform.shape[0] < waveform.shape[1]:
                # (C, T) -> (T, C)
                waveform = waveform.T

        audio_bytes = self._wav_to_bytes(waveform, sample_rate)
        return audio_bytes
