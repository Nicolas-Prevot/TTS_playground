import os
import torch
import torchaudio
import hydra
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from tts_core.base import BaseTTS

# --- CORRECT IMPORTS FOR v1.5.1 ---
# The function was renamed to 'load_model' in this version.
from fish_speech.models.text2semantic.inference import load_model as load_semantic_model, generate_long
# ----------------------------------

def load_codec_model(config_name, checkpoint_path, config_root, device="cuda"):
    """Load the Firefly GAN VQ model using Hydra."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    if not os.path.exists(config_root):
        raise FileNotFoundError(f"Config root path does not exist: {config_root}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    rel_config_path = os.path.relpath(os.path.abspath(config_root), current_dir)
    
    try:
        with initialize(version_base="1.3", config_path=rel_config_path):
            cfg = compose(config_name=config_name)
    except Exception as e:
        logger.error(f"Hydra initialization failed. Root: {config_root}. Error: {e}")
        raise

    model = instantiate(cfg)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        
    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model

class FishSpeech15Adapter(BaseTTS):
    def __init__(
        self,
        *,
        llama_checkpoint_dir: str = "checkpoints/fish-speech-1.5",
        codec_checkpoint_path: str = "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        decoder_config_name: str = "firefly_gan_vq",
        config_root_path: str = "../../../configs/fish-speech-1.5", 
        device: str = "cuda",
        half: bool = True,
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
        self.sr = 44100

    def load_model(self):
        # 1. Load Codec (Firefly GAN)
        target_config_path = os.path.abspath(self.config_root_path)
        
        logger.info(f"Loading Codec from {self.codec_path}...")
        self.codec_model = load_codec_model(
            config_name=self.decoder_cfg,
            checkpoint_path=self.codec_path,
            config_root=target_config_path,
            device=self.device,
        )
        self.sr = getattr(self.codec_model, "sample_rate", 44100)

        # 2. Load Semantic Model
        logger.info(f"Loading Semantic Model from {self.llama_dir}...")
        precision = torch.half if self.half else torch.bfloat16
        
        # CORRECTED CALL: Uses load_semantic_model (renamed from init_model)
        self.semantic_model, self.decode_one_token = load_semantic_model(
            checkpoint_path=self.llama_dir,
            device=self.device,
            precision=precision,
            compile=self.compile,
        )
        
        # 3. Patch KV Cache
        from types import MethodType
        device_to = self.device
        orig_setup = self.semantic_model.setup_caches
        
        def setup_and_move(model_self, max_batch_size, max_seq_len, dtype, *args, **kwargs):
            result = orig_setup(max_batch_size, max_seq_len, dtype, *args, **kwargs)
            for module in model_self.modules():
                if hasattr(module, "kv_cache"):
                    module.kv_cache = module.kv_cache.to(device_to)
            return result
        
        self.semantic_model.setup_caches = MethodType(setup_and_move, self.semantic_model)
        
        # Initialize cache
        self.semantic_model.setup_caches(
            max_batch_size=1,
            max_seq_len=self.semantic_model.config.max_seq_len,
            dtype=next(self.semantic_model.parameters()).dtype,
        )

        logger.success("Fish Speech 1.5 Loaded.")

    def clone_voice(self, ref_audio: str, ref_text: str = None):
        if not os.path.isfile(ref_audio):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

        wav, sr_orig = torchaudio.load(ref_audio)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr_orig != self.sr:
            wav = torchaudio.functional.resample(wav, sr_orig, self.sr)

        wav = wav.to(self.device).unsqueeze(0)
        lengths = torch.tensor([wav.size(-1)], device=self.device)
        
        indices = self.codec_model.encode(wav, lengths)
        if isinstance(indices, tuple): 
            indices = indices[0]
            
        #indices = indices.squeeze(0).long().to(self.device)
        indices = indices.squeeze(0).to(dtype=torch.int32, device=self.device)
        self.prompt_tokens = [indices]
        self.prompt_text = [ref_text] if ref_text else None
        
        logger.info(f"Voice cloned. Code shape: {indices.shape}")
        return True

    def synthesize(
        self,
        text: str,
        *,
        num_samples: int = 1,
        max_new_tokens: int = 0,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
        temperature: float = 0.7,
        chunk_length: int = 200, 
        seed: int = None,
    ) -> bytes:
        if self.codec_model is None:
            raise RuntimeError("Models not loaded.")
            
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

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
            raise RuntimeError("Generation produced no tokens.")

        codes = torch.cat(codes_accum, dim=1).to(self.device)
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)
        feature_lengths = torch.tensor([codes.shape[2]], device=self.device)

        out = self.codec_model.decode(codes, feature_lengths)
        if isinstance(out, tuple):
            audios = out[0]
        else:
            audios = out
            
        wav = audios[0, 0].cpu().detach().numpy()
        
        return self._wav_to_bytes(wav, self.sr)