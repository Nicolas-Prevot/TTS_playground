import os
import io
import torch
import torchaudio
import numpy as np
import soundfile as sf

import hydra
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger

from fish_speech.models.text2semantic.inference import init_model as init_semantic_model, generate_long

from tts_core.base import BaseTTS


def load_codec_model(config_name, checkpoint_path, config_root, device="cuda"):
    """
    Load the DAC codec model using Hydra configuration.
    
    Args:
        config_name: Name of the yaml file (without .yaml)
        checkpoint_path: Path to codec.pth
        config_root: Absolute path to the directory containing the config yaml
        device: cuda/cpu
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    # 1. Validate the path (using the absolute path)
    if not os.path.exists(config_root):
        logger.error(f"Config root path does not exist: {config_root}")
        raise FileNotFoundError(f"Config root path does not exist: {config_root}")

    # 2. Calculate relative path for Hydra
    # Hydra's 'initialize(config_path=...)' interprets the path relative to the 
    # Python file calling it (this file). We must calculate that relationship.
    try:
        # Get directory of this file: adapters/.../src/tts_adapter_openaudio_s1mini/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Calculate relative path from HERE to the config folder
        rel_config_path = os.path.relpath(os.path.abspath(config_root), current_dir)
        
        # Initialize Hydra with the relative path
        with initialize(version_base="1.3", config_path=rel_config_path):
            cfg = compose(config_name=config_name)
            
    except Exception as e:
        logger.error(f"Hydra initialization failed. \nConfig Root: {config_root}\nRelative: {rel_config_path}\nError: {e}")
        raise

    model = instantiate(cfg)
    state_dict = torch.load(
        checkpoint_path, map_location=device, mmap=True, weights_only=True
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()
    model.to(device)

    logger.info(f"[codec] Loaded DAC model from {checkpoint_path}. Loading state_dict: {result}")
    return model


class OpenAudioS1MiniAdapter(BaseTTS):
    """
    Adapter for Fish Audio's OpenAudio S1 Mini,
    using direct Python API calls (no subprocess).
    """
    def __init__(
        self,
        *,
        llama_checkpoint_dir: str = "checkpoints/openaudio-s1-mini",
        codec_checkpoint_path: str = "checkpoints/openaudio-s1-mini/codec.pth",
        decoder_config_name: str = "modded_dac_vq",
        config_root_path: str = "../../../configs/openaudio-s1-mini", 
        device: str = "cuda",
        half: bool = False,
        compile: bool = False,
    ):
        super().__init__()
        self.llama_dir = llama_checkpoint_dir
        self.codec_path = codec_checkpoint_path
        self.decoder_cfg = decoder_config_name
        self.config_root_path = config_root_path
        self.device = device
        self.half = half
        self.compile = compile

        self.codec_model = None
        self.semantic_model = None
        self.decode_one_token = None
        self.prompt_tokens = None
        self.prompt_text = None
        self.sr = None

    def load_model(self):
        """
        Load the DAC codec and DualARTransformer semantic model.
        """
        # 1. Load Codec
        # Pass the absolute path directly to load_codec_model; logic is handled there.
        target_config_path = os.path.abspath(self.config_root_path)

        self.codec_model = load_codec_model(
            config_name=self.decoder_cfg,
            checkpoint_path=self.codec_path,
            config_root=target_config_path,
            device=self.device,
        )
        self.sr = getattr(self.codec_model, "sample_rate", 24000)

        # 2. Load Semantic Model (LLaMA/Qwen based)
        precision = torch.half if self.half else torch.bfloat16
        self.semantic_model, self.decode_one_token = init_semantic_model(
            checkpoint_path=self.llama_dir,
            device=self.device,
            precision=precision,
            compile=self.compile,
        )
        
        # 3. Patch setup_caches to ensure KV cache moves to correct device 
        from types import MethodType
        device_to = self.device
        orig_setup = self.semantic_model.setup_caches
        
        def setup_and_move(model_self, max_batch_size, max_seq_len, dtype, *args, **kwargs):
            # 1) run the original CPU-only cache setup
            result = orig_setup(max_batch_size, max_seq_len, dtype, *args, **kwargs)
            # 2) relocate every kv_cache in every layer to our GPU
            for module in model_self.modules():
                if hasattr(module, "kv_cache"):
                    module.kv_cache = module.kv_cache.to(device_to)
            return result
        
        # replace the instance method
        self.semantic_model.setup_caches = MethodType(setup_and_move, self.semantic_model)

        # invoke it (will allocate *and* move to GPU)
        self.semantic_model.setup_caches(
            max_batch_size=1,
            max_seq_len=self.semantic_model.config.max_seq_len,
            dtype=next(self.semantic_model.parameters()).dtype,
        )

        logger.info("[semantic] Loaded DualARTransformer for text2semantic")

    def clone_voice(self, ref_audio: str, ref_text: str = None):
        if not os.path.isfile(ref_audio):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

        wav, sr_orig = torchaudio.load(ref_audio)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr_orig != self.sr:
            wav = torchaudio.functional.resample(wav, sr_orig, self.sr)

        wav = wav.to(self.device).unsqueeze(0)  # (1,1,T)
        lengths = torch.tensor([wav.size(-1)], device=self.device)
        
        # Encode audio to codes for prompting
        indices, lengths = self.codec_model.encode(wav, lengths)
        indices = indices.squeeze(0).long().to(self.device)
        self.prompt_tokens = [indices]

        if ref_text is not None:
            self.prompt_text = [ref_text]
        else:
            self.prompt_text = None

        logger.info("[clone_voice] Cached prompt tokens%s", 
                    " + text" if ref_text else "")
        return True

    def synthesize(
        self,
        text: str,
        *,
        num_samples: int          = 1,
        max_new_tokens: int       = 0,
        top_p: float              = 0.8,
        repetition_penalty: float = 1.1,
        temperature: float        = 0.8,
        chunk_length: int         = 0,
        seed: int                 = None,
    ) -> bytes:
        if self.codec_model is None or self.semantic_model is None:
            raise RuntimeError("Models not loaded; call load_model() first.")
        if self.prompt_tokens is None:
            raise RuntimeError("No reference cached; call clone_voice() first.")

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Generate semantic tokens (audio codes)
        generator = generate_long(
            model=self.semantic_model,
            device=self.device,
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

        codes_accum = []
        for resp in generator:
            if resp.action == "sample":
                codes_accum.append(resp.codes)
            elif resp.action == "next":
                break

        if not codes_accum:
            raise RuntimeError("Semantic generation produced no tokens.")

        codes = torch.cat(codes_accum, dim=1).to(self.device)

        # Decode codes to audio
        lengths = torch.tensor([codes.shape[1]], device=self.device)
        audios, _ = self.codec_model.decode(codes, lengths)
        wav = audios[0, 0].cpu().detach().numpy()  # shape (T,)

        return self._wav_to_bytes(wav, self.sr)

if __name__ == "__main__":
    # Simple self-test if run directly
    tts = OpenAudioS1MiniAdapter(
        llama_checkpoint_dir="checkpoints/openaudio-s1-mini",
        codec_checkpoint_path="checkpoints/openaudio-s1-mini/codec.pth",
        config_root_path="../../../configs/openaudio-s1-mini", 
        device="cuda" if torch.cuda.is_available() else "cpu",
        half=False
    )
    tts.load_model()
    tts.clone_voice("data/ref/basic_ref_en.wav", "Some call me nature.")
    wav = tts.synthesize("Hello world from S1 Mini.")
    with open("test_s1.wav", "wb") as f:
        f.write(wav)