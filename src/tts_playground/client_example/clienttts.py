import time
import base64
from pathlib import Path
from typing import Any, Dict, Optional
import httpx


class TTSClient:
    """Synchronous Python client for the TTS Playground API.

    Assumes the server exposes:
      - POST /v1/tts
      - GET  /v1/tasks/{task_id}
    """

    def __init__(self, base_url: str = "http://localhost:7000", *, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout)

    def _post(self, path: str, json: Dict[str, Any]) -> Dict[str, Any]:
        r = self.http.post(path, json=json)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str) -> Dict[str, Any]:
        r = self.http.get(path)
        r.raise_for_status()
        return r.json()

    def synth(
        self,
        adapter: str,
        init: Optional[Dict[str, Any]] = None,
        load_model: Optional[Dict[str, Any]] = None,
        clone_voice: Optional[Dict[str, Any]] = None,
        synthesize: Optional[Dict[str, Any]] = None,
        *,
        wait: bool = True,
        poll_interval: float = 1,
        timeout: Optional[float] = 120.0,
        download: bool = True,
        dest_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Start a TTS task with all parameters in one call. If `wait`, poll until done and optionally download the WAV.

        Payload structure:
          - adapter: str (required)
          - init: dict (optional, passed to adapter __init__)
          - load_model: dict (optional, passed to adapter.load_model(); {} if omitted)
          - clone_voice: dict (optional, passed to adapter.clone_voice())
          - synthesize: dict with "text": str (required) and "kwargs": dict (optional, passed to adapter.synthesize(text, **kwargs))

        Returns a dict with keys:
          - state: PENDING|STARTED|SUCCESS|FAILURE|TIMEOUT
          - task_id
          - (on SUCCESS) sr, wav_b64
        """
        if synthesize is None:
            synthesize = {}
        if "text" not in synthesize:
            raise ValueError("synthesize must contain 'text'")
        payload = {
            "adapter": adapter,
            "init": init or {},
            "load_model": load_model or {},
            "clone_voice": clone_voice or {},
            "synthesize": synthesize,
        }
        res = self._post("/v1/tts", payload)
        task_id = res["task_id"]

        if not wait:
            return {"state": "PENDING", "task_id": task_id}

        started = time.monotonic()
        while True:
            info = self.task_status(task_id)
            state = info["state"]
            if state == "SUCCESS":
                result: Dict[str, Any] = {"state": state, "task_id": task_id, **info}
                if download:
                    b64 = info["wav_b64"]
                    data = base64.b64decode(b64)
                    if dest_path is None:
                        dest = Path.cwd() / f"{task_id}.wav"
                    else:
                        dest = Path(dest_path)
                        if dest.is_dir():
                            dest = dest / f"{task_id}.wav"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                    result["local_path"] = str(dest)
                return result
            if state == "FAILURE":
                return {"state": state, "task_id": task_id, "error": info.get("error")}
            if timeout is not None and (time.monotonic() - started) > timeout:
                return {"state": "TIMEOUT", "task_id": task_id}
            time.sleep(poll_interval)

    def task_status(self, task_id: str) -> Dict[str, Any]:
        return self._get(f"/v1/tasks/{task_id}")

    # -------- helpers to pack inline voice refs --------
    @staticmethod
    def pack_file(path: str | Path, name: Optional[str] = None) -> Dict[str, str]:
        p = Path(path)
        return {
            "name": name or p.name,
            "b64": base64.b64encode(p.read_bytes()).decode("ascii"),
        }

    @staticmethod
    def pack_bytes(data: bytes, name: str) -> Dict[str, str]:
        return {"name": name, "b64": base64.b64encode(data).decode("ascii")}

    @staticmethod
    def pack_speaker_map(mapping: Dict[str | int, str | Path | Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        """
        Accepts { "1": "/path/a.wav", 2: {name,b64}, ... } and returns a pure blob map.
        """
        out: Dict[str, Dict[str, str]] = {}
        for k, v in mapping.items():
            sid = str(int(k))
            if isinstance(v, (str, Path)):
                out[sid] = TTSClient.pack_file(v)
            else:
                out[sid] = v  # assume already {name,b64}
        return out

    def close(self):
        self.http.close()














if __name__ == "__main__":
    ## Simple sync example
    client = TTSClient("http://localhost:7000")

    ## vibevoice A) Single-speaker with inline clone + custom local save path
    #en_ref = client.pack_file("data/ref/basic_ref_en.wav")
    #task = client.synth(
    #    adapter="vibevoicetts",
    #    init={},  # optional: e.g. {"model_id": "vibevoice/VibeVoice-7B", "cfg_scale": 1.3}
    #    load_model={},
    #    clone_voice={"ref_audio": en_ref},  # single reference
    #    synthesize={
    #        "text": "Hello from inline cloning.",
    #        "kwargs": {
    #            "generation_config": {"do_sample": False},
    #            "cfg_scale": 1.3,
    #            "ddpm_steps": 10,
    #        },
    #    },
    #    wait=True, download=True, timeout=300,
    #    dest_path=Path("gen_local/hello_inline.wav"),
    #)
    #print(task)

    ## vibevoice B) Multi-speaker: provide two voices; text must be labeled "Speaker N:"
    #fr_ref = client.pack_file("data/ref/fr/Ellie_Bishop_fr.wav")
    #script = (
    #    "Speaker 1: Hi, I'm the first.\n"
    #    "Speaker 2: Bonjour, je suis la seconde.\n"
    #    "Speaker 1: Nice to meet you!\n"
    #)
    #task = client.synth(
    #    adapter="vibevoicetts",
    #    load_model={},
    #    clone_voice={"voice_samples": [en_ref, fr_ref]},  # ordered list (Speaker 1 -> [0], etc.)
    #    synthesize={
    #        "text": script,
    #        "kwargs": {"cfg_scale": 1.2},
    #    },
    #    wait=True, download=True, timeout=300,
    #    dest_path=Path("gen_local/dialogue_list.wav"),
    #)
    #print(task)

    ## vibevoice C) Multi-speaker with explicit mapping per call (override in kwargs)
    #spk_map = client.pack_speaker_map({"1": en_ref, "2": fr_ref})
    #task = client.synth(
    #    adapter="vibevoicetts",
    #    load_model={},
    #    clone_voice={},  # no sticky clone
    #    synthesize={
    #        "text": script,
    #        "kwargs": {
    #            "speaker_voices": spk_map,  # per-call override
    #            "seed": 42,
    #            "generation_config": {"do_sample": True, "temperature": 0.8, "top_p": 0.9},
    #        },
    #    },
    #    wait=True, download=True, timeout=300,
    #    dest_path=Path("gen_local/dialogue_map.wav"),
    #)
    #print(task)

    # Example: IndexTTS2 one-shot
    en_ref = client.pack_file("data/ref/basic_ref_en.wav")
    emo_ref = client.pack_file("data/ref/emo_hate.wav")  # optional emotion ref
    task = client.synth(
        adapter="indextts2",
        init={  # optional, but specify if non-default
            "model_dir": "checkpoints/indextts2",
            "cfg_path": "checkpoints/indextts2/config.yaml",
            "use_fp16": False,
            "device": None,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.8,
            "max_text_tokens_per_segment": 120,
        },
        load_model={},
        clone_voice={"ref_audio": en_ref},
        synthesize={
            "text": "Hello from a Python client!",
            "kwargs": {
                "emo_audio_prompt": emo_ref,  # optional emotion ref (blob)
                "emo_alpha": 0.7,
                "temperature": 0.8,
                "top_p": 0.92,
            },
        },
        wait=True,
        download=True,
        timeout=300,
        dest_path="data/gen_api/indextts2.wav"
    )
    print("state:", task["state"])
    print("task_id:", task["task_id"])
    print("sr:", task["sr"])
    print("wav_b64:", task["wav_b64"][:30])

    client.close()