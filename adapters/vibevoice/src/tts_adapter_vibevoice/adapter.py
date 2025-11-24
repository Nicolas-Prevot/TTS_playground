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
    Adapter for the VibeVoice (community) models.
    - Supports: 1.5B / 7B checkpoints from HF (or a local dir)
    - Voice cloning via "prefill" (reference wavs)
    - Multi-speaker: parse 'Speaker N:' labels and align voices
    - LoRA adapter loading
    - flash_attention_2 (preferred on CUDA) with SDPA fallback
    - DDPM steps & CFG scale (classifier-free guidance)
    """

    def __init__(
        self,
        *,
        model_id: str = "vibevoice/VibeVoice-7B",        # HF repo or local dir
        device: Optional[str] = None,                    # "cuda" | "mps" | "cpu" | None (auto)
        torch_dtype: Optional[str] = None,               # "bfloat16" | "float16" | "float32" | None (auto)
        attn_implementation: Optional[str] = None,       # "flash_attention_2" | "sdpa" | None (auto)
        cfg_scale: float = 1.3,                          # CFG guidance
        ddpm_steps: int = 10,                            # DDPM inference steps
        is_prefill: bool = True,                         # enable voice cloning by default if refs exist
        generation_config: Optional[Dict[str, Any]] = None,  # e.g. {"do_sample": False}
        lora_checkpoint: Optional[str] = None,           # path to fine-tuned LoRA assets (optional)
        verbose: bool = False,
    ):
        super().__init__()
        self.model_id = model_id
        self.device_arg = (device or None)
        self.dtype_arg = (torch_dtype or None)
        self.attn_impl_arg = (attn_implementation or None)
        self.cfg_scale = cfg_scale
        self.ddpm_steps = ddpm_steps
        self.is_prefill_default = is_prefill
        self.gen_cfg_default = generation_config or {"do_sample": False}
        self.lora_checkpoint = lora_checkpoint
        self.verbose = verbose

        self.model: Optional[VibeVoiceForConditionalGenerationInference] = None
        self.processor: Optional[VibeVoiceProcessor] = None
        self.sr = 24000  # VibeVoice demos assume 24 kHz

        # Pre-fill storage
        self._voice_list_default: List[str] = []          # ordered list used when single-speaker or as fallback
        self._speaker_voices: Dict[str, str] = {}         # explicit mapping: {"1": "p1.wav", "2": "p2.wav"}

        # quiet HF if not verbose
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
            return {
                "float32": torch.float32, "fp32": torch.float32,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float16": torch.float16, "fp16": torch.float16,
            }[m]
        if device == "cuda":
            return torch.bfloat16
        return torch.float32  # MPS/CPU: fp32 for stability

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
        out = []
        for p in paths:
            if not osp.isfile(p):
                raise FileNotFoundError(f"Reference WAV not found: {p}")
            out.append(p)
        return out

    # --------------------------------------------------------------------- #
    # BaseTTS API
    # --------------------------------------------------------------------- #
    def load_model(self, **overrides):
        """
        Optional runtime overrides:
        - model_id, device, torch_dtype, attn_implementation, cfg_scale,
          ddpm_steps, is_prefill, generation_config, lora_checkpoint, verbose
        """
        for k, v in (overrides or {}).items():
            if hasattr(self, k):
                setattr(self, k, v)

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
                    self.model_id, torch_dtype=dtype, attn_implementation=attn_impl, device_map=None
                )
                return m.to("mps")
            if device == "cuda":
                return VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_id, torch_dtype=dtype, attn_implementation=attn_impl, device_map="cuda"
                )
            # CPU
            return VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_id, torch_dtype=dtype, attn_implementation=attn_impl, device_map="cpu"
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
          - For multi-speaker scripts, the order is determined by first appearance
            of "Speaker N:" in the text. We'll map N → file via `speaker_voices` if provided,
            otherwise pick from `voice_samples` by index (1-based), else fall back to
            the first default voice if present.
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
                    raise FileNotFoundError(f"Reference WAV not found for speaker {sid}: {v}")
                self._speaker_voices[sid] = v

        # must have at least *some* references to use prefill later
        return bool(self._voice_list_default or self._speaker_voices)

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        **kwargs,
    ) -> bytes:
        """
        Per-call overrides:
          - cfg_scale: float
          - ddpm_steps: int
          - is_prefill: bool
          - max_new_tokens: Optional[int]
          - generation_config: dict (e.g. {"do_sample": False, "temperature": 0.8, "top_p": 0.9})
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
                    raise FileNotFoundError(f"Reference WAV not found for speaker {sid}: {v}")
                tmp_speaker_map[sid] = v

        script = self._ensure_script_labels(text)
        speakers_in_text = self._first_appearance_speaker_order(script)
        if self.verbose:
            if speakers_in_text:
                print(f"[VibeVoice] detected speakers: {speakers_in_text}")

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
                    # no references at all → prefill must be disabled
                    pass
        else:
            # single-speaker text: use the default list (first item if present)
            if tmp_voice_list:
                voice_list_for_this_call = [tmp_voice_list[0]]

        # If no refs resolved, prefill cannot be used
        use_prefill = is_prefill and bool(voice_list_for_this_call)

        # Prepare processor inputs
        inputs = self.processor(
            text=[script],                                      # batch of size 1
            voice_samples=[voice_list_for_this_call] if voice_list_for_this_call else [[]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move tensors to model device
        target = self.model.device if hasattr(self.model, "device") else next(self.model.parameters()).device
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(target)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            cfg_scale=cfg_scale,
            tokenizer=self.processor.tokenizer,
            generation_config=gen_cfg,
            verbose=self.verbose,
            is_prefill=use_prefill,
        )

        if not getattr(outputs, "speech_outputs", None) or outputs.speech_outputs[0] is None:
            raise RuntimeError("No audio generated by VibeVoice.")

        # (Optional) small telemetry
        try:
            input_tokens = int(inputs["input_ids"].shape[1])
            output_tokens = int(outputs.sequences.shape[1])
            if self.verbose:
                print(f"[VibeVoice] tokens: prefilling={input_tokens}, generated={output_tokens - input_tokens}, total={output_tokens}")
        except Exception:
            pass

        # Extract 1-D float waveform in [-1, 1]
        wav = outputs.speech_outputs[0]
        if isinstance(wav, torch.Tensor):
            # Expect shape [B=1, T] or [T]; convert to 1-D CPU float
            if wav.ndim == 2 and wav.shape[0] == 1:
                wav = wav[0]
            wav = wav.detach().to("cpu").float().numpy()

        return self._wav_to_bytes(wav, self.sr)


# ------------------------------------------------------------------------- #
# Examples
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    os.makedirs("data/gen", exist_ok=True)

    tts = VibeVoiceAdapter(
        model_id="vibevoice/VibeVoice-7B",   # or "vibevoice/VibeVoice-1.5B"
        device=None,                         # auto: cuda > mps > cpu
        torch_dtype=None,                    # auto: bf16 on CUDA, f32 otherwise
        attn_implementation=None,            # auto: flash_attention_2 on CUDA else sdpa
        cfg_scale=1.3,
        ddpm_steps=10,
        is_prefill=True,
        generation_config={"do_sample": False},
        verbose=True,
    )
    tts.load_model()

    # ---------- A) Single-speaker with prefill ----------
    ref_wav = "data/ref/basic_ref_en.wav"
    tts.clone_voice(ref_audio=ref_wav)
    out = tts.synthesize("Hello! This is VibeVoice via TTS_PLAYGROUND.")
    open("data/gen/vibevoice_single_prefill.wav", "wb").write(out)

    # ---------- B) Single-speaker without prefill + mild sampling ----------
    out = tts.synthesize(
        "Now speaking without voice cloning and with mild sampling.",
        is_prefill=False,  # disable even if a ref voice exists
        generation_config={"do_sample": True, "temperature": 0.8, "top_p": 0.9},
        seed=42,
    )
    open("data/gen/vibevoice_single_noprefill.wav", "wb").write(out)

    # ---------- C) Multi-speaker using an explicit speaker→voice map ----------
    # Text must be labeled "Speaker 1:", "Speaker 2:", ...
    script = (
        "Speaker 1: Hi! I'm the first speaker.\n"
        "Speaker 2: And I'm the second speaker.\n"
        "Speaker 1: Great to meet you!\n"
    )
    spk_map = {
        "1": "data/ref/basic_ref_en.wav",
        "2": "data/ref/fr/Ellie_Bishop_fr.wav",
    }
    tts.clone_voice(speaker_voices=spk_map)  # set once
    out = tts.synthesize(script, cfg_scale=1.2)
    open("data/gen/vibevoice_multi_map.wav", "wb").write(out)

    # ---------- D) Multi-speaker using a voice list (1-based order) ----------
    # If you don't want to provide a dict mapping, you can pass a list.
    # The adapter will map: Speaker 1 -> list[0], Speaker 2 -> list[1], etc.
    voices = [
        "data/ref/basic_ref_en.wav",
        "data/ref/fr/Ellie_Bishop_fr.wav",
    ]
    tts.clone_voice(voice_samples=voices)
    out = tts.synthesize(script)  # automatically aligns by first appearance
    open("data/gen/vibevoice_multi_list.wav", "wb").write(out)