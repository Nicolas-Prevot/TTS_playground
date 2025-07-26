import os
import shlex
import json
import subprocess
from pathlib import Path
from datetime import datetime
from loguru import logger
from typing import Any


UNIFIED_SCRIPT = "src/tts_playground/inference_demo.py"

CMD_PREFIX: dict[str, str] = {
    "F5-TTS":           "uv sync --extra f5 && uv run python",
    "ChatterboxTTS":    "uv sync --extra chatterbox && uv run python",
    "Index-TTS":        "conda run -n index-tts python",
    "Kokoro82M":        "uv sync --extra kokoro && uv run python",
    "KyutaiTTS":        "uv sync --extra kyutai && uv run python",
    "OpenAudioS1-mini": "uv sync --extra fishaudio && uv run python",
    "HiggsAudio":       "uv sync --extra higgsaudio && uv run python",
}

MODEL_BEHAVIOR: dict[str, dict[str, Any]] = {
    "F5-TTS": {
        "clone": True,
        "need_ref_text": True,
        "lang_supported": ["en", "fr"],
    },
    "ChatterboxTTS": {
        "clone": True,
        "need_ref_text": False,
        "lang_supported": ["en", "fr"],
    },
    "Index-TTS": {
        "clone": True,
        "need_ref_text": False,
        "lang_supported": ["en", "fr"],
    },
    "Kokoro82M": {
        "clone": False,
        "voices": {
            "en": [
                {"lang_code": "a", "voice": "af_bella"},
                {"lang_code": "a", "voice": "af_sarah"},
                {"lang_code": "a", "voice": "am_fenrir"},
                {"lang_code": "b", "voice": "bf_alice"},
                {"lang_code": "b", "voice": "bm_lewis"},
            ],
            "fr": [
                {"lang_code": "f", "voice": "ff_siwis"},
            ]
        },
    },
    "KyutaiTTS": {
        "clone": False,
        "voices": {
            "en": [
                {"voice_sample": "vctk/p228_023.wav"},
                {"voice_sample": "vctk/p303_023.wav"},
                {"voice_sample": "vctk/p304_023.wav"},
                {"voice_sample": "vctk/p311_023.wav"},
            ],
            "fr": [
                {"voice_sample": "cml-tts/fr/12080_11650_000047-0001.wav"},
                {"voice_sample": "cml-tts/fr/1591_1028_000108-0004.wav"},
                {"voice_sample": "unmute-prod-website/fabieng-enhanced-v2.wav"},
                {"voice_sample": "unmute-prod-website/degaulle-2.wav"},
            ]
        },
    },
    "OpenAudioS1-mini": {
        "clone": True,
        "need_ref_text": True,
        "lang_supported": ["en", "fr"],
    },
    "HiggsAudio": {
        "clone": True,
        "need_ref_text": True,
        "lang_supported": ["en", "fr"],
    },
}

REF: dict[str, list[str]] = {
    "en": ["data/ref/basic_ref_en", "data/ref/en/Elon_Musk", "data/ref/en/Liam_Neeson", "data/ref/en/Liam_Neeson_short", "data/ref/en/Liam_Neeson_long",
           "data/ref/en/Lucy_Chen", "data/ref/en/Lucy_Chen_long", "data/ref/en/Lucy_Chen_2", "data/ref/en/Morty", "data/ref/en/Rick", "data/ref/en/The_Rock",
           "data/ref/en/Tom_Holland", "data/ref/en/belinda", "data/ref/en/bigbang_amy", "data/ref/en/bigbang_sheldon", "data/ref/en/broom_salesman",
           "data/ref/en/chadwick", "data/ref/en/en_man", "data/ref/en/en_woman", "data/ref/en/fiftyshades_anna", "data/ref/en/mabel", "data/ref/en/shrek_donkey",
           "data/ref/en/shrek_fiona", "data/ref/en/shreck_shreck", "data/ref/en/vex", "data/ref/en/Lucifer", "data/ref/en/Lucifer_2"],


    "fr": ["data/ref/fr/Ellie_Bishop_fr", "data/ref/fr/Emmanuel_Macron", "data/ref/fr/Jean_Dujardin", "data/ref/fr/Louis_de_Funes", "data/ref/fr/Ziva_fr"],
}

PROMPTS: dict[str, list[str]] = {
    "en": [
        "data/gen/prompt_en_1.txt",
    ],
    "fr": [
        "data/gen/prompt_fr_1.txt",
    ],
}


class OrchestratorAudio:
    def __init__(
        self,
        ref=REF,
        prompts=PROMPTS,
        unified_script=UNIFIED_SCRIPT,
        cmd_prefix=CMD_PREFIX,
        model_behavior=MODEL_BEHAVIOR,
        progress_file: str = "data/gen/progress.jsonl",
    ):
        self.ref = ref
        self.prompts = prompts
        self.unified_script = unified_script
        self.cmd_prefix = cmd_prefix
        self.model_behavior = model_behavior

        self.progress_file = progress_file
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)

    def build_unified_cmd(
        self,
        prefix: str,
        model: str,
        text: str,
        out_path: str,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        clone_args: dict[str, Any] | None = None,
        synth_args: dict[str, Any] | None = None,
        init_args: dict[str, Any] | None = None,
        load_args: dict[str, Any] | None = None
    ) -> str:

        parts = [
            shlex.quote(self.unified_script),
            "--model", shlex.quote(model),
            "--text", shlex.quote(text),
            "--out", shlex.quote(out_path),
        ]
        if ref_audio:
            parts += ["--ref_audio", shlex.quote(ref_audio)]
        if ref_text:
            parts += ["--ref_text", shlex.quote(ref_text)]
        for flag, data in [("--clone_args", clone_args), ("--synth_args", synth_args), ("--init_args", init_args), ("--load_args", load_args)]:
            if data is not None:
                parts += [flag, shlex.quote(json.dumps(data, separators=(",", ":")))] # shlex.quote(json.dumps(data))   f"'{str(json.dumps(data)).replace(" ", "")}'"

        cmd = f"{prefix} {' '.join(parts)}"
        return cmd
    
    def _record_progress(
        self,
        task: str,
        model: str,
        lang: str,
        ref_or_voice: str,
        prompt_idx: int,
        status: str,
        cmd: str
    ):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "model": model,
            "lang": lang,
            "ref_or_voice": ref_or_voice,
            "prompt_index": prompt_idx,
            "status": status,
            "cmd": cmd,
        }
        with open(self.progress_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _get_clone_models(self, lang: str) -> list[str]:
        return [
            m for m, beh in self.model_behavior.items()
            if beh.get("clone", False) and lang in beh.get("lang_supported", [])
        ]

    def _get_synth_models(self, lang: str) -> list[str]:
        return [
            m for m, beh in self.model_behavior.items()
            if not beh.get("clone", False)
            and lang in beh.get("voices", {})
            and beh["voices"][lang]
        ]
    
    def _ensure_dir(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def run_clone_for_lang(self, lang: str, output_dir: str="data/gen"):
        logger.info(f"Voice cloning {lang} on {len(self.ref.get(lang, []))} reference voices with the models: {self._get_clone_models(lang)}")
        for model in self._get_clone_models(lang):
            beh = self.model_behavior[model]
            prefix = self.cmd_prefix[model]
            need_ref_text = beh.get("need_ref_text", False)

            for ref_base in self.ref.get(lang, []):
                ref_audio = f"{ref_base}.wav"
                ref_text  = f"{ref_base}.txt" if need_ref_text else None
                for i, prompt in enumerate(self.prompts.get(lang, []), start=1):
                    out_path = f"{output_dir}/{model}/{lang}/{Path(ref_base).name}_{i}.wav"
                    self._ensure_dir(out_path)

                    cmd = self.build_unified_cmd(
                        prefix=prefix,
                        model=model,
                        text=prompt,
                        out_path=out_path,
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                    )

                    logger.info(f"[CLONE][{lang}][{model}]{ref_base} → {out_path}")
                    try:
                        subprocess.run(cmd, shell=True, check=True)
                        status = "SUCCESS"
                    except subprocess.CalledProcessError as e:
                        status = f"ERROR: {e.returncode}"  # capture error code
                        logger.error(f"Generation failed for {model} {lang} prompt#{i}")
                    finally:
                        self._record_progress(
                            task="clone",
                            model=model,
                            lang=lang,
                            ref_or_voice=Path(ref_base).name,
                            prompt_idx=i,
                            status=status,
                            cmd=cmd
                        )

    def run_synth_for_lang(self, lang: str, output_dir: str="data/gen"):
        logger.info(f"Synthesize {lang} with the models: {self._get_synth_models(lang)}")
        for model in self._get_synth_models(lang):
            beh = self.model_behavior[model]
            prefix = self.cmd_prefix[model]
            for voice_cfg in beh["voices"][lang]:
                if "voice" in voice_cfg:
                    voice_name = voice_cfg["voice"]
                else:
                    voice_name = Path(voice_cfg["voice_sample"]).stem

                for i, prompt in enumerate(self.prompts.get(lang, []), start=1):
                    out_path = f"{output_dir}/{model}/{lang}/{voice_name}_{i}.wav"
                    self._ensure_dir(out_path)

                    cmd = self.build_unified_cmd(
                        prefix=prefix,
                        model=model,
                        text=prompt,
                        out_path=out_path,
                        clone_args=voice_cfg
                    )

                    logger.info(f"[SYNTH][{lang}][{model}][{voice_name}] → {out_path}")
                    logger.info(cmd)
                    try:
                        subprocess.run(cmd, shell=True, check=True)
                        status = "SUCCESS"
                    except subprocess.CalledProcessError as e:
                        status = f"ERROR: {e.returncode}"
                        logger.error(f"Synthesis failed for {model} {voice_name} {lang} prompt#{i}")
                    finally:
                        self._record_progress(
                            task="synth",
                            model=model,
                            lang=lang,
                            ref_or_voice=voice_name,
                            prompt_idx=i,
                            status=status,
                            cmd=cmd
                        )


if __name__ == "__main__":
    orchestrator = OrchestratorAudio(ref=REF, prompts=PROMPTS)
    #orchestrator.run_clone_for_lang("en")
    #orchestrator.run_clone_for_lang("fr")
    #orchestrator.run_synth_for_lang("en")
    orchestrator.run_synth_for_lang("fr")
    

"""
# TEMP : demo that works
for model, cmd_prefix in MODELS.items():
    print(model, cmd_prefix)


    ref_audio = "data/ref/basic_ref_en.wav"
    ref_text = "data/ref/basic_ref_en.txt"
    output_file = f"data/gen/{model}_basic_en.wav"
    cmd = (
        f"{cmd_prefix} --model {model} "
        f"--ref_audio {ref_audio} "
        f"--ref_text {ref_text} "
        f'--text "I don\'t really care what you call me. I\'ve been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring." '
        f"--out {output_file}"
    )

    subprocess.run(cmd, shell=True, check=True)
"""