import os
import os.path as osp
import re
from typing import Any, Dict, List, Optional, Union

import torch
from transformers.utils import logging as hf_logging

from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.lora_loading import load_lora_assets

from tts_core.base import BaseTTS


_SPK_REGEX = re.compile(r"^\s*Speaker\s+(\d+)\s*:", flags=re.IGNORECASE | re.MULTILINE)


class VibeVoiceAdapter(BaseTTS):
    """
    Adapter for VibeVoice (community) models.

    Supports:
    - 1.5B / 7B checkpoints (HF repo id or local directory)
    - Voice cloning via "prefill" (reference wavs)
    - Multi-speaker scripts: parse "Speaker N:" labels and align voices
    - Optional LoRA adapter loading
    - flash_attention_2 (preferred on CUDA) with SDPA fallback
    - DDPM steps & CFG scale
    """

    def __init__(
        self,
        *,
        model_id: str = "vibevoice/VibeVoice-7B",
        device: Optional[str] = None,  # "cuda" | "mps" | "cpu" | None (auto)
        torch_dtype: Optional[str] = None,  # "bfloat16" | "float16" | "float32" | None (auto)
        attn_implementation: Optional[str] = None,  # "flash_attention_2" | "sdpa" | None (auto)
        cfg_scale: float = 1.3,
        ddpm_steps: int = 10,
        is_prefill: bool = True,
        generation_config: Optional[Dict[str, Any]] = None,
        lora_checkpoint: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__()

        # Public-ish config
        self.model_id = model_id
        self.cfg_scale = cfg_scale
        self.ddpm_steps = ddpm_steps
        self.is_prefill_default = is_prefill
        self.gen_cfg_default = generation_config or {"do_sample": False}
        self.lora_checkpoint = lora_checkpoint
        self.verbose = verbose

        # Internal config
        self.device_arg = device or None
        self.dtype_arg = torch_dtype or None
        self.attn_impl_arg = attn_implementation or None

        self.model: Optional[VibeVoiceForConditionalGenerationInference] = None
        self.processor: Optional[VibeVoiceProcessor] = None
        self.sr = 24000  # VibeVoice demos assume 24kHz

        # Prefill storage
        self._voice_list_default: List[str] = []          # ordered list for default/fallback
        self._speaker_voices: Dict[str, str] = {}         # explicit mapping: {"1": "p1.wav", ...}

        # Quiet HF if not verbose
        if not self.verbose:
            hf_logging.set_verbosity_error()

    # --------------------------------------------------------------------- #
    # utilities
    # --------------------------------------------------------------------- #
    def _pick_device(self) -> str:
        if self.device_arg:
            d = self.device_arg.lower()
            if d == "mpx":  # common typo
                return "mps"
            return d
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _pick_dtype(self, device: str) -> torch.dtype:
        if self.dtype_arg:
            m = self.dtype_arg.lower()
            mapping = {
                "float32": torch.float32, "fp32": torch.float32,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float16": torch.float16, "fp16": torch.float16,
            }
            if m not in mapping:
                raise ValueError(f"Unknown torch_dtype '{self.dtype_arg}'. Use float32/float16/bfloat16.")
            return mapping[m]

        if device == "cuda":
            # Prefer bf16 if supported, else fp16
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
            return torch.float16

        # MPS/CPU: fp32 for stability
        return torch.float32

    def _pick_attn(self, device: str) -> str:
        if self.attn_impl_arg:
            return self.attn_impl_arg
        return "flash_attention_2" if device == "cuda" else "sdpa"

    @staticmethod
    def _first_appearance_speaker_order(text: str) -> List[str]:
        """Return speakers (as strings) in order of first appearance: ['1','2', ...]."""
        seen, order = set(), []
        for m in _SPK_REGEX.finditer(text or ""):
            sid = str(int(m.group(1)))  # normalize to e.g. '1'
            if sid not in seen:
                order.append(sid)
                seen.add(sid)
        return order

    def _ensure_script_labels(self, text: str) -> str:
        """If no 'Speaker N:' labels exist, convert to a Speaker 1 script."""
        if _SPK_REGEX.search(text or ""):
            return text
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            raise ValueError("Empty text.")
        if len(lines) == 1:
            return f"Speaker 1: {lines[0]}"
        return "\n".join(f"Speaker 1: {ln}" for ln in lines)

    @staticmethod
    def _validate_files(paths: List[str]) -> List[str]:
        out: List[str] = []
        for p in paths:
            if not osp.isfile(p):
                raise FileNotFoundError(f"Reference file not found: {p}")
            out.append(p)
        return out

    # --------------------------------------------------------------------- #
    # BaseTTS API
    # --------------------------------------------------------------------- #
    def load_model(self, **overrides):
        """
        Optional runtime overrides:
          - model_id, device, torch_dtype, attn_implementation,
            cfg_scale, ddpm_steps, is_prefill, generation_config,
            lora_checkpoint, verbose
        """
        # Apply overrides with explicit mapping (public API keys -> internal attrs)
        if overrides:
            if "model_id" in overrides:
                self.model_id = overrides["model_id"]
            if "device" in overrides:
                self.device_arg = overrides["device"]
            if "torch_dtype" in overrides:
                self.dtype_arg = overrides["torch_dtype"]
            if "attn_implementation" in overrides:
                self.attn_impl_arg = overrides["attn_implementation"]
            if "cfg_scale" in overrides:
                self.cfg_scale = float(overrides["cfg_scale"])
            if "ddpm_steps" in overrides:
                self.ddpm_steps = int(overrides["ddpm_steps"])
            if "is_prefill" in overrides:
                self.is_prefill_default = bool(overrides["is_prefill"])
            if "generation_config" in overrides and overrides["generation_config"] is not None:
                self.gen_cfg_default = dict(overrides["generation_config"])
            if "lora_checkpoint" in overrides:
                self.lora_checkpoint = overrides["lora_checkpoint"]
            if "verbose" in overrides:
                self.verbose = bool(overrides["verbose"])

        # Update HF verbosity based on current verbose flag
        if self.verbose:
            hf_logging.set_verbosity_info()
        else:
            hf_logging.set_verbosity_error()

        device = self._pick_device()
        dtype = self._pick_dtype(device)
        attn_primary = self._pick_attn(device)

        if self.verbose:
            print(f"[VibeVoice] load: model_id={self.model_id}, device={device}, dtype={dtype}, attn={attn_primary}")

        # Processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_id)

        # Model with FA2→SDPA fallback
        def _load(attn_impl: str):
            if device == "mps":
                m = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    attn_implementation=attn_impl,
                    device_map=None,
                )
                return m.to("mps")
            if device == "cuda":
                return VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    attn_implementation=attn_impl,
                    device_map="cuda",
                )
            # CPU
            return VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
                device_map="cpu",
            )

        try:
            self.model = _load(attn_primary)
        except Exception as e:
            if attn_primary == "flash_attention_2":
                print(f"[VibeVoice] flash_attention_2 failed ({type(e).__name__}: {e}); falling back to SDPA.")
                self.model = _load("sdpa")
            else:
                raise

        # Optional LoRA
        if self.lora_checkpoint:
            report = load_lora_assets(self.model, self.lora_checkpoint)
            if self.verbose:
                print(f"[VibeVoice] LoRA loaded: {report}")

        # Diffusion steps & eval
        self.model.set_ddpm_inference_steps(num_steps=int(self.ddpm_steps))
        self.model.eval()

    def clone_voice(
        self,
        ref_audio: Optional[Union[str, List[str]]] = None,
        *,
        voice_samples: Optional[List[str]] = None,
        speaker_voices: Optional[Dict[Union[str, int], str]] = None,
    ):
        """
        Set reference voices to enable prefill (voice cloning).

        Options:
          - ref_audio="a.wav"                         (single reference, single speaker)
          - voice_samples=["a.wav","b.wav",...]       (ordered default list; used for single-speaker or as fallback)
          - speaker_voices={"1": "a.wav", 2: "b.wav"} (explicit mapping for multi-speaker scripts)

        Notes:
          - For multi-speaker scripts, order is determined by first appearance of "Speaker N:".
            We'll map N → file via `speaker_voices` if provided, otherwise pick from `voice_samples`
            by index (1-based), else fall back to the first default voice if present.
        """
        # default list
        if ref_audio is not None:
            voice_samples = [ref_audio] if isinstance(ref_audio, str) else list(ref_audio)
        if voice_samples:
            self._voice_list_default = self._validate_files(voice_samples)

        # explicit mapping
        self._speaker_voices = {}
        if speaker_voices:
            for k, v in speaker_voices.items():
                sid = str(int(k))
                if not osp.isfile(v):
                    raise FileNotFoundError(f"Reference file not found for speaker {sid}: {v}")
                self._speaker_voices[sid] = v

        return bool(self._voice_list_default or self._speaker_voices)

    @torch.no_grad()
    def synthesize(self, text: str, **kwargs) -> bytes:
        """
        Per-call overrides:
          - cfg_scale: float
          - ddpm_steps: int
          - is_prefill: bool
          - max_new_tokens: Optional[int]
          - generation_config: dict (e.g. {"do_sample": True, "temperature": 0.8})
          - seed: Optional[int]
          - speaker_voices: Optional[Dict[Union[str,int], str]]  # per-call override
          - voice_samples: Optional[List[str]]                   # per-call override
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # seed (optional)
        if "seed" in kwargs and kwargs["seed"] is not None:
            seed = int(kwargs["seed"])
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # per-call overrides
        cfg_scale = float(kwargs.get("cfg_scale", self.cfg_scale))
        is_prefill = bool(kwargs.get("is_prefill", self.is_prefill_default))
        max_new_tokens = kwargs.get("max_new_tokens", None)

        ddpm_steps = kwargs.get("ddpm_steps", None)
        if ddpm_steps is not None:
            self.model.set_ddpm_inference_steps(num_steps=int(ddpm_steps))

        gen_cfg = dict(self.gen_cfg_default)
        gen_cfg.update(kwargs.get("generation_config", {}))

        # voice refs (allow per-call override)
        tmp_voice_list = self._voice_list_default[:]
        tmp_speaker_map = dict(self._speaker_voices)

        if "voice_samples" in kwargs and kwargs["voice_samples"]:
            tmp_voice_list = self._validate_files(list(kwargs["voice_samples"]))

        if "speaker_voices" in kwargs and kwargs["speaker_voices"]:
            tmp_speaker_map = {}
            for k, v in kwargs["speaker_voices"].items():
                sid = str(int(k))
                if not osp.isfile(v):
                    raise FileNotFoundError(f"Reference file not found for speaker {sid}: {v}")
                tmp_speaker_map[sid] = v

        script = self._ensure_script_labels(text)

        speakers_in_text = self._first_appearance_speaker_order(script)
        if self.verbose and speakers_in_text:
            print(f"[VibeVoice] detected speakers: {speakers_in_text}")

        # Build ordered voice list matching Speaker 1..N
        voice_list_for_this_call: List[str] = []
        if speakers_in_text:
            for idx, sid in enumerate(speakers_in_text, start=1):
                if sid in tmp_speaker_map:
                    voice_list_for_this_call.append(tmp_speaker_map[sid])
                elif tmp_voice_list and (idx - 1) < len(tmp_voice_list):
                    voice_list_for_this_call.append(tmp_voice_list[idx - 1])  # 1-based mapping
                elif tmp_voice_list:
                    voice_list_for_this_call.append(tmp_voice_list[0])        # fallback to first ref
                else:
                    # no references at all
                    pass
        else:
            # single-speaker text: use first default voice if present
            if tmp_voice_list:
                voice_list_for_this_call = [tmp_voice_list[0]]

        # Prefill is only possible if we have references AND user wants it
        use_prefill = is_prefill and bool(voice_list_for_this_call)

        # IMPORTANT:
        # If prefill is OFF, pass voice_samples=None so processor does not build speech tensors.
        voice_samples_arg = [voice_list_for_this_call] if use_prefill else None

        inputs = self.processor(
            text=[script],                # batch size 1
            voice_samples=voice_samples_arg,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move tensors to model device
        target = getattr(self.model, "device", None)
        if target is None:
            target = next(self.model.parameters()).device

        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(target)

        # Generate (DO NOT pass is_prefill to generate; upstream doesn't accept it reliably)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            cfg_scale=cfg_scale,
            tokenizer=self.processor.tokenizer,
            generation_config=gen_cfg,
            verbose=self.verbose,
        )

        if not getattr(outputs, "speech_outputs", None) or outputs.speech_outputs[0] is None:
            raise RuntimeError("No audio generated by VibeVoice.")

        # Extract 1-D float waveform in [-1, 1]
        wav = outputs.speech_outputs[0]
        if isinstance(wav, torch.Tensor):
            if wav.ndim == 2 and wav.shape[0] == 1:
                wav = wav[0]
            wav = wav.detach().to("cpu").float().numpy()

        return self._wav_to_bytes(wav, self.sr)
