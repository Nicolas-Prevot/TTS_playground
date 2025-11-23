import json, os, subprocess, threading, uuid, sys
from pathlib import Path
from typing import Any, Dict


VENVS_DIR = Path(os.getenv("RUNNER_VENVS_DIR", "/workspace/venvs")).resolve()
VENVS_DIR.mkdir(parents=True, exist_ok=True)


ADAPTERS: dict[str, tuple[str, str, str]] = {
    "higgsaudio": ("higgsaudio", "tts_playground.adapters.HiggsAudioTTS", "HiggsAudioAdapter"),
    "indextts2": ("indextts2", "tts_playground.adapters.IndexTTS2", "IndexTTS2Adapter"),
    "vibevoicetts": ("vibevoicetts", "tts_playground.adapters.VibeVoiceTTS", "VibeVoiceAdapter"),
    "f5tts": ("f5", "tts_playground.adapters.F5_TTS", "F5TTSAdapter"),
    "chatterbox": ("chatterbox", "tts_playground.adapters.ChatterboxTTS", "ChatterboxTTSAdapter"),
    "openaudios1mini": ("fishaudio", "tts_playground.adapters.OpenAudioS1MiniTTS", "OpenAudioS1MiniAdapter"),
    "kokoro": ("kokoro", "tts_playground.adapters.kokoroTTS", "KokoroTTSAdapter"),
    "kyutai": ("kyutai", "tts_playground.adapters.KyutaiTTS", "KyutaiTTSAdapter"),
}


class _Future:
    def __init__(self):
        self._ev = threading.Event()
        self._res: Any = None

    def set(self, v: Any):
        self._res = v
        self._ev.set()

    def get(self) -> Any:
        self._ev.wait()
        return self._res


class RunnerProc:
    def __init__(self, adapter: str):
        self.adapter = adapter
        self.idle_secs = int(os.getenv("IDLE_SECS", "180"))
        self.exit_on_idle = str(os.getenv("EXIT_ON_IDLE", "1")).lower() not in ("", "0", "false", "no")
        self.proc = None
        self._waiter = threading.Thread(target=self._read_loop, daemon=True)
        self._futures: dict[str, _Future] = {}

    def _ensure_proc(self):
        if self.proc and self.proc.poll() is None:
            return
        extra, mod, cls = ADAPTERS[self.adapter]
        envdir = VENVS_DIR / self.adapter
        exit_flag = "1" if self.exit_on_idle else "0"
        #cmd = (
        #    f'uv venv "{envdir}" && '
        #    f'UV_PROJECT_ENVIRONMENT="{envdir}" uv sync --locked --extra {extra} && '
        #    f'"{envdir}/bin/python" -m tts_playground.runtime.adapter_runner ' # f'UV_PROJECT_ENVIRONMENT="{envdir}" uv run python -m tts_playground.runtime.adapter_runner '
        #    f'--adapter {self.adapter} --module {mod} --cls {cls} --idle {self.idle_secs} --exit-on-idle {exit_flag}'
        #)
        cmd = (
            f'set -e; '  # NEW: Exit on any error
            f'echo "[RUNNER {self.adapter}] Starting venv setup"; '
            f'uv venv --seed "{envdir}" && echo "[RUNNER {self.adapter}] Venv created"; ' # for kokoro
            #f'uv venv "{envdir}" && echo "[RUNNER {self.adapter}] Venv created"; '
            f'UV_PROJECT_ENVIRONMENT="{envdir}" uv sync --locked --extra {extra} && echo "[RUNNER {self.adapter}] Sync complete"; '
            f'echo "[RUNNER {self.adapter}] Starting adapter_runner"; '
            f'"{envdir}/bin/python" -m tts_playground.runtime.adapter_runner '
            f'--adapter {self.adapter} --module {mod} --cls {cls} --idle {self.idle_secs} --exit-on-idle {exit_flag} '
            f'&& echo "[RUNNER {self.adapter}] Adapter runner exited normally"'
        )
        env = dict(os.environ)
        src_path = str(Path.cwd() / "src") # v2
        env.update({
            "HF_HOME": "/cache/hf",
            "HF_HUB_CACHE": "/cache/hf",
            "PYTHONPATH": f"{src_path}:{env.get('PYTHONPATH', '')}", # v2
        })
        # self.proc = subprocess.Popen(
        #     ["bash", "-lc", cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env
        # )
        # self._waiter = threading.Thread(target=self._read_loop, daemon=True)
        # self._waiter.start()
        self.proc = subprocess.Popen(
            ["bash", "-lc", cmd], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True, 
            bufsize=1, 
            env=env
        )
        self._waiter = threading.Thread(target=self._read_loop, daemon=True)
        self._waiter.start()
        stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        stderr_thread.start()
        if self.proc.poll() is not None:
            sys.stderr.write(f"[RUNNER {self.adapter}] Process exited immediately with code {self.proc.returncode}\n")

    def _read_loop(self):
        if not self.proc or not self.proc.stdout:
            return
        for line in iter(self.proc.stdout.readline, ''):
            line = line.rstrip('\r\n')
            if not line:
                continue
            sys.stdout.write(f"[DEBUG READ {self.adapter}] Raw line (len={len(line)}): {repr(line[:100])}{'...' if len(line) > 100 else ''}\n")
            try:
                msg = json.loads(line)
                rid = msg.get("id", "")
                fut = self._futures.pop(rid, None)
                if fut:
                    sys.stdout.write(f"[DEBUG SET {self.adapter}] Set fut for rid={rid}, ok={msg.get('ok')}\n")
                    fut.set(msg)
                else:
                    sys.stdout.write(f"[DEBUG NOFUT {self.adapter}] Parsed msg for unknown rid={rid}\n")
            except json.JSONDecodeError as e:
                sys.stdout.write(f"[RUNNER_STDOUT {self.adapter}]: {line}\n")
                sys.stdout.write(f"[DEBUG JSONERR {self.adapter}] Decode error: {str(e)}\n")
            except Exception as e:
                sys.stdout.write(f"[DEBUG PARSEERR {self.adapter}] Unexpected error on line: {str(e)}\n")
        for rid, fut in list(self._futures.items()):
            sys.stdout.write(f"[DEBUG REMAIN {self.adapter}] Setting error for remaining rid={rid}\n")
            fut.set({"ok": False, "error": "runner exited"})
            self._futures.pop(rid, None)

    def _stderr_loop(self):
        if not self.proc or not self.proc.stderr:
            return
        for line in iter(self.proc.stderr.readline, ''):
            line = line.strip()
            if line:
                sys.stderr.write(f"[RUNNER_STDERR {self.adapter}]: {line}\n")

    def _req(self, method: str, params: Dict[str, Any]) -> Any:
        self._ensure_proc()
        assert self.proc and self.proc.stdin
        rid = uuid.uuid4().hex
        fut = _Future()
        self._futures[rid] = fut
        payload = {"id": rid, "method": method, "params": params}
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        resp = fut.get()
        if not resp.get("ok", False):
            raise RuntimeError(resp.get("error", "runner error"))
        return resp.get("result")

    def run_all(
        self,
        init: Dict[str, Any],
        load_model: Dict[str, Any],
        clone: Dict[str, Any],
        text: str,
        adapter_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        params = {
            "init": init,
            "load_model": load_model,
            "clone_voice": clone,
            "synthesize": {"text": text, "kwargs": adapter_args},
        }
        return self._req("run", params)


class RunnerManager:
    def __init__(self):
        self._procs: dict[str, RunnerProc] = {}
        self._lock = threading.Lock()

    def get(self, adapter: str) -> RunnerProc:
        if adapter not in ADAPTERS:
            raise KeyError(f"Unknown adapter: {adapter}")
        with self._lock:
            rp = self._procs.get(adapter)
            if rp is None:
                rp = RunnerProc(adapter)
                self._procs[adapter] = rp
            return rp


manager = RunnerManager()