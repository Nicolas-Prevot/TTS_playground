from importlib.resources import files
from cached_path import cached_path
from omegaconf import OmegaConf
from hydra.utils import get_class
from f5_tts.infer.utils_infer import (
    device as default_device,
    load_vocoder,
    load_model as f5_load_model,
    preprocess_ref_audio_text,
    infer_process,
    mel_spec_type as default_mel_spec_type,
    nfe_step as default_nfe_step,
    cfg_strength as default_cfg_strength,
    sway_sampling_coef as default_sway_sampling_coef,
    speed as default_speed,
    fix_duration as default_fix_duration,
    target_rms as default_target_rms,
    cross_fade_duration as default_cross_fade_duration,
    remove_silence_for_generated_wav
)

from tts_playground.base import BaseTTS


class F5TTSAdapter(BaseTTS):
    """
    Adapter for SWivid’s F5-TTS (and E2-TTS) local inference.
    """
    def __init__(
        self,
        *,
        model_name: str = "F5TTS_v1_Base", # E2TTS_Base / E2TTS_Small / F5TTS_Base / F5TTS_Small / F5TTS_v1_Base (cf f5_tts/configs)
        model_cfg_path: str = None,
        ckpt_file: str = "",
        vocab_file: str = "",
        vocoder_name: str = default_mel_spec_type, # vocos / bigvgan
        load_vocoder_from_local: bool = False,
        vocoder_local_path: str = "",
        device: str = default_device,
        hf_cache_dir: str = None,
    ):
        super().__init__()

        self.model_name         = model_name
        self.model_cfg_path     = model_cfg_path or str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
        self.ckpt_file          = ckpt_file
        self.vocab_file         = vocab_file
        self.vocoder_name       = vocoder_name
        self.load_vocoder_local = load_vocoder_from_local
        self.vocoder_local_path = vocoder_local_path
        self.device             = device
        self.hf_cache_dir       = hf_cache_dir

        self.model = None
        self.vocoder = None
        self.sr = None

        self._cached_ref_audio = None
        self._cached_ref_text  = None
    
    def load_model(self):
        """Instantiate vocoder + diffusion TTS model"""
        # Vocoder
        if self.vocoder_name.lower()=="bigvgan":
            import bigvgan
            repo = "nvidia/bigvgan_v2_24khz_100band_256x"
            voc = bigvgan.BigVGAN.from_pretrained(
                repo,
                use_cuda_kernel=False,
                cache_dir=self.hf_cache_dir
            )
            voc.remove_weight_norm()
            self.vocoder = voc.eval().to(self.device)
        else:
            self.vocoder = load_vocoder(
                vocoder_name=self.vocoder_name,
                is_local=self.load_vocoder_local,
                local_path=self.vocoder_local_path,
                device=self.device,
                hf_cache_dir=self.hf_cache_dir,
            )

        # TTS model
        model_cfg = OmegaConf.load(self.model_cfg_path)
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arch= model_cfg.model.arch
        
        repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

        # if self.model_name != "F5TTS_Base":
        #     assert self.vocoder_name == model_cfg.model.mel_spec.mel_spec_type

        if self.model_name == "F5TTS_Base":
            if self.vocoder_name == "vocos":
                ckpt_step = 1200000
            elif self.vocoder_name == "bigvgan":
                self.model_name = "F5TTS_Base_bigvgan"
                ckpt_type = "pt"
        elif self.model_name == "E2TTS_Base":
            repo_name = "E2-TTS"
            ckpt_step = 1200000

        if not self.ckpt_file:
            self.ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{self.model_name}/model_{ckpt_step}.{ckpt_type}", cache_dir=self.hf_cache_dir))

        print(f"[F5TTSAdapter] Using model {self.model_name} from {self.ckpt_file}")
        self.model = f5_load_model(
            model_cls,
            model_arch,
            self.ckpt_file,
            mel_spec_type=self.vocoder_name,
            vocab_file=self.vocab_file,
            device=self.device,
        )
        self.sr = 24000
    
    def synthesize(self, 
                   text: str = None,
                   *,
                   speed: float    = None,
                   nfe_step: int   = None,
                   cfg_strength: float = None,
                   sway_sampling_coef: float = None,
                   cross_fade_duration: float = None,
                   target_rms: float = None,
                   fix_duration: float = None,
                   device: str     = None,
        ) -> bytes:
        if self.model is None or self.vocoder is None:
            raise RuntimeError("Model not loaded; call load_model() first.")
        
        if not self._ref_audio_prep:
            raise ValueError("No reference audio: call clone_voice() first.")

        params = {
            "speed":                speed                or default_speed,
            "nfe_step":             nfe_step             or default_nfe_step,
            "cfg_strength":         cfg_strength         or default_cfg_strength,
            "sway_sampling_coef":   sway_sampling_coef   or default_sway_sampling_coef,
            "cross_fade_duration":  cross_fade_duration  or default_cross_fade_duration,
            "target_rms":           target_rms           or default_target_rms,
            "fix_duration":         fix_duration         or default_fix_duration,
            "device":               device               or self.device,
        }
        
        wav_np, sr, _spec = infer_process(
            self._ref_audio_prep,
            self._ref_text_prep,
            text,
            self.model,
            self.vocoder,
            mel_spec_type = self.vocoder_name,
            **params
        )
        return self._wav_to_bytes(wav_np, sr)
    
    def clone_voice(self, ref_audio: str = None, ref_text: str = None):
        self._ref_audio_prep, self._ref_text_prep = preprocess_ref_audio_text(ref_audio, ref_text)
        return True
    
    def remove_silence_for_generated_wav(self, filename):
        remove_silence_for_generated_wav(filename)


if __name__ == "__main__":

    # vocos   + E2TTS_Small No / E2TTS_Base Yes / F5TTS_Base Yes / F5TTS_Small No / F5TTS_v1_Base Yes
    # bigvgan + E2TTS_Small No / E2TTS_Base No  / F5TTS_Base Yes / F5TTS_Small No / F5TTS_v1_Base No
    tts = F5TTSAdapter(
        model_name="F5TTS_v1_Base",
        vocoder_name="vocos",  #  bigvgan  vocos
    )

    tts.load_model()

    tts.clone_voice(
        "data/ref/basic_ref_en.wav",
        "Some call me nature, others call me mother nature."
    )
    audio_bytes = tts.synthesize(
        text="I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring.",
        speed=1.0,)

    with open("data/gen/test.wav", "wb") as f:
        f.write(audio_bytes)

    tts.remove_silence_for_generated_wav("data/gen/test.wav")