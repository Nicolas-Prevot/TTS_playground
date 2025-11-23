import os
import numpy as np
import soundfile as sf
from indextts.infer import IndexTTS
import sys

_this_dir = os.path.dirname(__file__)
_src_root = os.path.abspath(os.path.join(
    _this_dir,
    os.pardir,
    os.pardir 
))
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from tts_playground.base import BaseTTS


class IndexTTSAdapter(BaseTTS):

    def __init__(
        self,
        *,
        model_dir: str = "checkpoints",
        cfg_path: str = "checkpoints/config.yaml",
        use_fp16: bool = True,
        device: str = None,
        use_cuda_kernel: bool = False,
        fast: bool = False,

        do_sample: bool = True,
        top_p: float = 0.8,
        top_k: int = 30,
        temperature: float = 1.0,
        num_beams: int = 3,
        length_penalty: float = 0.0,
        repetition_penalty: float = 10.0,
        max_mel_tokens: int = 600,

        max_text_tokens_per_segment: int = 100,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.cfg_path = cfg_path
        self.use_fp16 = use_fp16
        self.device = device
        self.use_cuda_kernel = use_cuda_kernel
        self.fast = fast

        self.gen_kwargs = {
            "do_sample": do_sample,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "max_mel_tokens": max_mel_tokens,
        }
        self.fast_kwargs = {
            "max_text_tokens_per_segment": max_text_tokens_per_segment,
        }

        self.model: IndexTTS = None
        self.sr: int = None
        self._audio_prompt: str = None

    def load_model(self):
        self.model = IndexTTS(
            cfg_path=self.cfg_path,
            model_dir=self.model_dir,
            use_fp16=self.use_fp16,
            device=self.device,
            use_cuda_kernel=self.use_cuda_kernel
        )

        self.sr = 24000

    def clone_voice(self, ref_audio: str):
        if not os.path.isfile(ref_audio):
            raise FileNotFoundError(f"Voice sample not found: {ref_audio}")
        self._audio_prompt = ref_audio
        self.model.cache_cond_mel = None
        return True

    def synthesize(self, text: str, **kwargs) -> bytes:

        if self.model is None or self.sr is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self._audio_prompt is None:
            raise ValueError("No voice cloned. Call clone_voice() first.")


        params = {**self.gen_kwargs, **self.fast_kwargs}
        for k, v in kwargs.items():
            if k in params:
                params[k] = v
        use_fast = kwargs.get("fast", self.fast)

        if use_fast:
            out = self.model.infer_fast(
                audio_prompt=self._audio_prompt,
                text=text,
                output_path=None,
                **params
            )
        else:
            out = self.model.infer(
                audio_prompt=self._audio_prompt,
                text=text,
                output_path=None,
                **params
            )

        if isinstance(out, tuple):
            sr, wav_np = out
        else:
            sr, wav_np = self.sr, sf.read(out)[0]

        if isinstance(wav_np, np.ndarray):
            return self._wav_to_bytes(wav_np, sr)
        else:
            wav = out.cpu().numpy().T if hasattr(out, "cpu") else np.array(out)
            return self._wav_to_bytes(wav, sr)


if __name__ == "__main__":

    tts = IndexTTSAdapter(
        model_dir="checkpoints/indextts",
        cfg_path="checkpoints/indextts/config.yaml",
        use_fp16=True,
        device="cuda",                      # or "cpu"
        use_cuda_kernel=True,              # set True if you built the custom CUDA ops
        fast=False,                         # standard (higher-quality) inference
    )

    tts.load_model()

    tts.clone_voice(ref_audio="data/ref/basic_ref_en.wav") # "data/ref/basic_ref_en.wav"

    wav_bytes = tts.synthesize(
        "Hello, this is a demo of IndexTTS zero-shot voice cloning!",
        do_sample=True,           # sampling-based decoding
        top_p=0.9,                # nucleus sampling
        temperature=0.7,          # more conservative sampling
        num_beams=5,              # beam search width
        fast=False                # set True for faster but lower-quality output
    )

    with open("data/gen/index_output_demok.wav", "wb") as f:
        f.write(wav_bytes)







