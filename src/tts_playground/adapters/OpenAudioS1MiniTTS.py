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

from fish_speech.models.text2semantic.inference import init_model as init_semantic_model, generate_long, GenerateResponse

from tts_playground.base import BaseTTS


def load_codec_model(config_name, checkpoint_path, device="cuda"):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../../configs/openaudio-s1-mini"):
        cfg = compose(config_name=config_name)

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
        device: str = "cuda",
        half: bool = False,
        compile: bool = False,
    ):
        super().__init__()
        self.llama_dir = llama_checkpoint_dir
        self.codec_path = codec_checkpoint_path
        self.decoder_cfg = decoder_config_name
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
        self.codec_model = load_codec_model(
            config_name=self.decoder_cfg,
            checkpoint_path=self.codec_path,
            device=self.device,
        )
        self.sr = getattr(self.codec_model, "sample_rate", 24000)

        precision = torch.half if self.half else torch.bfloat16
        self.semantic_model, self.decode_one_token = init_semantic_model(
            checkpoint_path=self.llama_dir,
            device=self.device,
            precision=precision,
            compile=self.compile,
        )
        self.semantic_model.setup_caches(
            max_batch_size=1,
            max_seq_len=self.semantic_model.config.max_seq_len,
            dtype=next(self.semantic_model.parameters()).dtype,
        )
        logger.info("[semantic] Loaded DualARTransformer for text2semantic")

    def clone_voice(self, voice_sample: str, ref_text: str = None):
        if not os.path.isfile(voice_sample):
            raise FileNotFoundError(f"Reference audio not found: {voice_sample}")

        wav, sr_orig = torchaudio.load(voice_sample)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr_orig != self.sr:
            wav = torchaudio.functional.resample(wav, sr_orig, self.sr)

        wav = wav.to(self.device).unsqueeze(0)  # (1,1,T)
        lengths = torch.tensor([wav.size(-1)], device=self.device)
        indices, lengths = self.codec_model.encode(wav, lengths)
        indices = indices.squeeze(0).cpu().long()
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

        lengths = torch.tensor([codes.shape[1]], device=self.device)
        audios, _ = self.codec_model.decode(codes, lengths)
        wav = audios[0, 0].cpu().detach().numpy()  # shape (T,)

        buf = io.BytesIO()
        sf.write(buf, wav, self.sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()

if __name__ == "__main__":

    tts = OpenAudioS1MiniAdapter(
        llama_checkpoint_dir="checkpoints/openaudio-s1-mini",
        codec_checkpoint_path="checkpoints/openaudio-s1-mini/codec.pth",
        decoder_config_name="modded_dac_vq",
        device="cuda",      # or "cpu"
        half=False,         # use float32
    )
    tts.load_model()
    tts.clone_voice("data/ref/basic_ref_en.wav",
                    ref_text="Some call me nature, others call me mother nature.")

    english_bytes = tts.synthesize(
        "(shouting)I don't really care what you call me. (shouting)I've been a silent spectator, "
        "watching species evolve, empires rise and fall. (shouting)But always remember, "
        "I am mighty and enduring, (laughing) Ha,ha,ha!",
        max_new_tokens=0,
        top_p=0.9,
        repetition_penalty=1.1,
        temperature=0.8
    )
    with open("data/gen/test_s1_eng2.wav", "wb") as f:
        f.write(english_bytes)