import os
import inspect
import torch
import torchaudio

import hydra
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from loguru import logger

from fish_speech.models.text2semantic.inference import (
    init_model as init_semantic_model,
    generate_long,
)

from tts_core.base import BaseTTS


def _pick_sample_rate(codec_model, default: int = 24000) -> int:
    # Prefer explicit attribute (DAC often exposes sample_rate)
    sr = getattr(codec_model, "sample_rate", None)
    if isinstance(sr, (int, float)):
        return int(sr)

    # Some Fish-Speech models expose spec_transform.sample_rate
    st = getattr(codec_model, "spec_transform", None)
    sr = getattr(st, "sample_rate", None) if st is not None else None
    if isinstance(sr, (int, float)):
        return int(sr)

    return int(default)


def load_codec_model(
    config_name: str,
    checkpoint_path: str,
    config_root: str,
    device: torch.device,
):
    """
    Load the DAC codec model using a Hydra config from a filesystem directory.

    Args:
        config_name: YAML name without .yaml (e.g. "modded_dac_vq")
        checkpoint_path: path to codec.pth
        config_root: directory containing the YAML
        device: torch.device
    """
    # Hydra is a singleton; clear to allow re-init in long-lived processes
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    config_root = os.path.abspath(config_root)
    if not os.path.isdir(config_root):
        raise FileNotFoundError(f"Config root path does not exist: {config_root}")

    try:
        with initialize_config_dir(version_base="1.3", config_dir=config_root):
            cfg = compose(config_name=config_name)
    except Exception as e:
        logger.error(
            "Hydra initialization failed.\n"
            f"  config_root: {config_root}\n"
            f"  config_name: {config_name}\n"
            f"  error: {e}"
        )
        raise

    model = instantiate(cfg)

    # torch.load() compatibility (some older torch builds may not support weights_only/mmap)
    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            mmap=True,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Some checkpoints may prefix "generator."
    if isinstance(state_dict, dict) and any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    # assign=True is optional and may not exist on older torch; keep compatible
    try:
        result = model.load_state_dict(state_dict, strict=False, assign=True)
    except TypeError:
        result = model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(device)

    logger.info(f"[codec] Loaded codec from {checkpoint_path}. load_state_dict: {result}")
    return model


class OpenAudioS1MiniAdapter(BaseTTS):
    """
    Adapter for OpenAudio S1 Mini via Fish-Speech Python APIs.

    Pipeline:
      text -> semantic codes (Dual-AR Transformer)
      semantic codes -> waveform (DAC codec)
    """

    def __init__(
        self,
        *,
        llama_checkpoint_dir: str = "checkpoints/openaudio-s1-mini",
        codec_checkpoint_path: str = "checkpoints/openaudio-s1-mini/codec.pth",
        decoder_config_name: str = "modded_dac_vq",
        config_root_path: str = "configs/openaudio-s1-mini",
        device: str = "cuda",
        half: bool = False,
        compile: bool = False,
    ):
        super().__init__()
        self.llama_dir = llama_checkpoint_dir
        self.codec_path = codec_checkpoint_path
        self.decoder_cfg = decoder_config_name
        self.config_root_path = config_root_path
        self.device_str = device
        self.half = half
        self.compile = compile

        self._device: torch.device | None = None
        self.codec_model = None
        self.semantic_model = None
        self.decode_one_token = None

        self.prompt_tokens = None
        self.prompt_text = None
        self.sr = None

    def _resolve_device(self) -> torch.device:
        dev = torch.device(self.device_str)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' requested but CUDA is not available. "
                "Set device='cpu' or configure CUDA correctly."
            )
        return dev

    def _pick_precision(self, device: torch.device) -> torch.dtype:
        # half=True means fp16 (matches upstream `--half`)
        if self.half:
            return torch.float16

        # Prefer bf16 on CUDA if supported; otherwise fall back to fp16
        is_bf16_ok = False
        if device.type == "cuda":
            is_bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        return torch.bfloat16 if (device.type == "cuda" and is_bf16_ok) else (
            torch.float16 if device.type == "cuda" else torch.float32
        )

    def _ensure_kv_cache_on_device(self, device: torch.device) -> None:
        """
        Some Fish-Speech versions may allocate KV cache on CPU.
        Safely move it to the target device without assuming kv_cache.to() exists.
        """
        if self.semantic_model is None:
            return

        for module in self.semantic_model.modules():
            kv = getattr(module, "kv_cache", None)
            if kv is None:
                continue

            # If kv_cache implements .to(), use it
            if hasattr(kv, "to"):
                try:
                    module.kv_cache = kv.to(device)
                    kv = module.kv_cache
                except Exception:
                    # fall back to moving inner tensors
                    pass

            # Move known tensor fields
            for name in ("k_cache", "v_cache"):
                t = getattr(kv, name, None)
                if isinstance(t, torch.Tensor) and t.device != device:
                    setattr(kv, name, t.to(device))

    def load_model(self):
        """
        Load codec (DAC) and semantic model (Dual-AR Transformer).
        """
        device = self._resolve_device()
        self._device = device

        # 1) Codec
        self.codec_model = load_codec_model(
            config_name=self.decoder_cfg,
            checkpoint_path=self.codec_path,
            config_root=self.config_root_path,
            device=device,
        )
        self.sr = _pick_sample_rate(self.codec_model, default=24000)

        # 2) Semantic model
        precision = self._pick_precision(device)
        self.semantic_model, self.decode_one_token = init_semantic_model(
            checkpoint_path=self.llama_dir,
            device=device,
            precision=precision,
            compile=self.compile,
        )

        # 3) Setup caches (if available)
        if hasattr(self.semantic_model, "setup_caches"):
            try:
                sig = inspect.signature(self.semantic_model.setup_caches)
                kwargs = dict(
                    max_batch_size=1,
                    max_seq_len=self.semantic_model.config.max_seq_len,
                    dtype=next(self.semantic_model.parameters()).dtype,
                )
                if "device" in sig.parameters:
                    kwargs["device"] = device
                self.semantic_model.setup_caches(**kwargs)
            except Exception as e:
                logger.warning(f"[semantic] setup_caches failed (continuing): {e}")

        # Ensure KV cache is on target device if it exists
        self._ensure_kv_cache_on_device(device)

        logger.info("[semantic] Loaded DualARTransformer for text2semantic")

    def clone_voice(self, ref_audio: str, ref_text: str | None = None):
        if self.codec_model is None:
            raise RuntimeError("Codec not loaded; call load_model() first.")

        if not os.path.isfile(ref_audio):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

        device = self._device or self._resolve_device()

        wav, sr_orig = torchaudio.load(ref_audio)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr_orig != self.sr:
            wav = torchaudio.functional.resample(wav, sr_orig, self.sr)

        # (B=1, C=1, T)
        wav = wav.to(device).unsqueeze(0)
        lengths = torch.tensor([wav.size(-1)], device=device, dtype=torch.long)

        # Encode prompt tokens
        indices, _ = self.codec_model.encode(wav, lengths)

        # Keep shape consistent with upstream: (Q, T) as a single prompt item in a list
        prompt = indices[0].long().to(device)
        self.prompt_tokens = [prompt]

        self.prompt_text = [ref_text] if ref_text is not None else None

        logger.info("[clone_voice] Cached prompt tokens%s", " + text" if ref_text else "")
        return True

    def synthesize(
        self,
        text: str,
        *,
        num_samples: int = 1,
        max_new_tokens: int = 0,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        temperature: float = 0.8,
        chunk_length: int = 0,
        seed: int | None = None,
    ) -> bytes:
        if self.codec_model is None or self.semantic_model is None:
            raise RuntimeError("Models not loaded; call load_model() first.")
        if self.prompt_tokens is None:
            raise RuntimeError("No reference cached; call clone_voice() first.")
        if num_samples != 1:
            raise NotImplementedError(
                "num_samples != 1 is not implemented in this adapter yet."
            )

        device = self._device or self._resolve_device()

        if seed is not None:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)

        generator = generate_long(
            model=self.semantic_model,
            device=device,
            decode_one_token=self.decode_one_token,
            text=text,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            compile=self.compile,
            iterative_prompt=(chunk_length > 0),
            chunk_length=chunk_length,
            prompt_text=self.prompt_text,
            prompt_tokens=self.prompt_tokens,
        )

        codes_accum: list[torch.Tensor] = []
        for resp in generator:
            if resp.action == "sample":
                codes_accum.append(resp.codes)
            elif resp.action == "next":
                break

        if not codes_accum:
            raise RuntimeError("Semantic generation produced no tokens.")

        # Concatenate along time dimension
        codes = torch.cat(codes_accum, dim=1).to(device).long()

        # Make batch explicit for decoder
        if codes.ndim == 2:  # (Q, T)
            codes_b = codes.unsqueeze(0)  # (1, Q, T)
            lengths = torch.tensor([codes.shape[1]], device=device, dtype=torch.long)
        elif codes.ndim == 3:  # (B, Q, T)
            codes_b = codes
            lengths = torch.full(
                (codes_b.shape[0],),
                fill_value=codes_b.shape[2],
                device=device,
                dtype=torch.long,
            )
        else:
            raise RuntimeError(f"Unexpected codes shape: {tuple(codes.shape)}")

        audios, _ = self.codec_model.decode(codes_b, lengths)
        wav = audios[0, 0].detach().float().cpu().numpy()

        return self._wav_to_bytes(wav, self.sr)


if __name__ == "__main__":
    tts = OpenAudioS1MiniAdapter(
        llama_checkpoint_dir="checkpoints/openaudio-s1-mini",
        codec_checkpoint_path="checkpoints/openaudio-s1-mini/codec.pth",
        config_root_path="configs/openaudio-s1-mini",
        device="cuda" if torch.cuda.is_available() else "cpu",
        half=True,
    )
    tts.load_model()
    tts.clone_voice("data/ref/basic_ref_en.wav", "Some call me nature.")
    wav = tts.synthesize("Hello world from S1 Mini.")
    with open("test_s1.wav", "wb") as f:
        f.write(wav)
