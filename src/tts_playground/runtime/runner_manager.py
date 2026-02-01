import os
import subprocess
import threading
import uuid
import json
import sys
import base64
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger


ADAPTER_REGISTRY = {
    "indextts2": ("indextts2", "tts_adapter_indextts2.adapter", "IndexTTS2Adapter"),
    "chatterbox": ("chatterbox", "tts_adapter_chatterbox.adapter", "ChatterboxTTSAdapter"),
    "f5tts": ("f5tts", "tts_adapter_f5tts.adapter", "F5TTSAdapter"),
    "higgsaudio": ("higgsaudio", "tts_adapter_higgsaudio.adapter", "HiggsAudioAdapter"),
    "kokoro": ("kokoro", "tts_adapter_kokoro.adapter", "KokoroTTSAdapter"),
    "kyutai": ("kyutai", "tts_adapter_kyutai.adapter", "KyutaiTTSAdapter"),
    "openaudios1mini": ("openaudio_s1mini", "tts_adapter_openaudio_s1mini.adapter", "OpenAudioS1MiniAdapter"),
    "vibevoicetts": ("vibevoice", "tts_adapter_vibevoice.adapter", "VibeVoiceAdapter"),
    "fishspeech15": ("fish_speech_1_5", "tts_adapter_fish_speech_1_5.adapter", "FishSpeech15Adapter"),
    "dia2": ("dia2", "tts_adapter_dia2.adapter", "Dia2Adapter"),
    "qwen3tts": ("qwen3tts", "tts_adapter_qwen3tts.adapter", "Qwen3TTSAdapter"),
}

ADAPTERS_ROOT = Path(os.getenv("ADAPTERS_DIR", "./adapters")).resolve()
VENV_ROOT = Path(os.getenv("RUNNER_VENVS_DIR", "./runner_envs")).resolve()
REPO_ROOT = ADAPTERS_ROOT.parent

REQUIRED_INTERNAL_PATHS = {
    "higgsaudio": ["src/tts_adapter_higgsaudio/audio_processing"],
}
REQUIRED_EXTERNAL_PATHS = {
    "dia2": ["external/dia2/pyproject.toml"],
}


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


class RunnerProc:
    def __init__(self, adapter_name: str):
        self.adapter_name = adapter_name
        self.config = ADAPTER_REGISTRY[adapter_name]
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._futures: Dict[str, Any] = {}

        self.idle_secs = int(os.getenv("IDLE_SECS", "180"))
        self.exit_on_idle = os.getenv("EXIT_ON_IDLE", "1") == "1"
        self.request_timeout = int(os.getenv("RUNNER_REQUEST_TIMEOUT", "1800"))

    def _ensure_environment(self, adapter_dir: Path, venv_dir: Path):
        """
        Checks if the .venv exists. If not, runs 'uv sync' to create it.
        """
        if not adapter_dir.exists():
            raise RuntimeError(f"[{self.adapter_name}] Adapter directory not found: {adapter_dir}")

        # 1. Check Internal Paths (Vendored code inside adapter)
        reqs_internal = REQUIRED_INTERNAL_PATHS.get(self.adapter_name, [])
        missing_internal = [p for p in reqs_internal if not (adapter_dir / p).exists()]
        if missing_internal:
            missing_str = "\n  - " + "\n  - ".join(missing_internal)
            raise RuntimeError(
                f"[{self.adapter_name}] Missing required INTERNAL paths:{missing_str}\n"
                f"Please copy these files into the adapter directory as per the README."
            )

        # 2. Check External Paths (Sibling repos/dependencies)
        # These are checked relative to the Project Root (parent of 'adapters')
        reqs_external = REQUIRED_EXTERNAL_PATHS.get(self.adapter_name, [])
        missing_external = [p for p in reqs_external if not (REPO_ROOT / p).exists()]
        if missing_external:
            missing_str = "\n  - " + "\n  - ".join(missing_external)
            raise RuntimeError(
                f"[{self.adapter_name}] Missing required EXTERNAL paths:{missing_str}\n"
                f"This adapter requires an external repository clone.\n"
                f"Run the setup commands in 'adapters/{self.adapter_name}/README.md'."
            )

        python_exe = _venv_python(venv_dir)
        if python_exe.exists():
            return
        
        logger.info(f"[{self.adapter_name}] Creating env at {venv_dir} with uv sync...")

        env = os.environ.copy()
        env["UV_PROJECT_ENVIRONMENT"] = str(venv_dir)
        env.setdefault("UV_CACHE_DIR", "/uv_cache")
        cmd = ["uv", "sync", "--frozen", "--no-dev", "--no-editable"]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(adapter_dir),
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            logger.debug(f"[{self.adapter_name}] uv sync stdout:\n{result.stdout}")
            logger.debug(f"[{self.adapter_name}] uv sync stderr:\n{result.stderr}")
            logger.success(f"[{self.adapter_name}] Environment created successfully: {python_exe}")
        except subprocess.CalledProcessError as e:
            msg = (
                f"[{self.adapter_name}] uv sync failed (code={e.returncode}).\n"
                f"--- stdout ---\n{e.stdout}\n"
                f"--- stderr ---\n{e.stderr}\n"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if not python_exe.exists():
            raise RuntimeError(f"[{self.adapter_name}] Env created but python missing: {python_exe}")

    def _start_process(self):
        folder_name, mod_path, cls_name = self.config
        adapter_dir = ADAPTERS_ROOT / folder_name
        venv_dir = VENV_ROOT / folder_name

        self._ensure_environment(adapter_dir, venv_dir)

        python_exe = _venv_python(venv_dir)

        runner_script = Path(__file__).parent / "adapter_runner.py"

        cmd = [
            str(python_exe),
            str(runner_script),
            "--module", mod_path,
            "--cls", cls_name,
            "--idle", str(self.idle_secs),
            "--exit-on-idle", "1" if self.exit_on_idle else "0",
        ]

        logger.info(f"[{self.adapter_name}] Spawning: {' '.join(cmd)}")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        if self.adapter_name == "dia2":
            ext_root = str(REPO_ROOT / "external" / "dia2")
            env["PYTHONPATH"] = ext_root + os.pathsep + env.get("PYTHONPATH", "")
        
        runner_cwd = str(adapter_dir)
        if self.adapter_name == "indextts2":
            runner_cwd = str(REPO_ROOT)

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=runner_cwd,  # str(adapter_dir),
        )

        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._stderr_loop, daemon=True).start()

    def stop(self):
        """Explicitly stop the runner process to free memory."""
        with self._lock:
            if self.proc:
                if self.proc.poll() is None:
                    logger.info(f"[{self.adapter_name}] Stopping runner to free resources...")
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"[{self.adapter_name}] Force killing runner...")
                        self.proc.kill()
                self.proc = None

    def _read_loop(self):
        while self.proc and self.proc.poll() is None:
            try:
                line = self.proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue

                msg = json.loads(line)
                rid = msg.get("id")

                with self._lock:
                    if rid in self._futures:
                        self._futures[rid]["result"] = msg
                        self._futures[rid]["event"].set()
            except json.JSONDecodeError:
                logger.debug(f"[{self.adapter_name} RAW] {line}")
            except Exception as e:
                logger.error(f"[{self.adapter_name}] Read loop error: {e}")
                break

    def _stderr_loop(self):
        while self.proc and self.proc.poll() is None:
            try:
                line = self.proc.stderr.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    logger.info(f"[{self.adapter_name}] {line}")
            except Exception:
                break

    def _is_blob(self, item: Any) -> bool:
        return isinstance(item, dict) and "b64" in item and "name" in item

    def _stage_files(self, data: Any, temp_dir: Path) -> Any:
        """Recursively find B64 blobs, write to disk, replace with Path str."""
        if isinstance(data, dict):
            if self._is_blob(data):
                safe_name = "".join(c for c in data["name"] if c.isalnum() or c in "._-")
                file_path = temp_dir / safe_name
                file_path.write_bytes(base64.b64decode(data["b64"]))
                return str(file_path)
            return {k: self._stage_files(v, temp_dir) for k, v in data.items()}

        if isinstance(data, list):
            return [self._stage_files(i, temp_dir) for i in data]

        return data

    def _req(self, method: str, params: Dict[str, Any]) -> Any:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start_process()

        rid = uuid.uuid4().hex
        event = threading.Event()

        with self._lock:
            self._futures[rid] = {"event": event, "result": None}

        try:
            payload = {"id": rid, "method": method, "params": params}
            try:
                self.proc.stdin.write(json.dumps(payload) + "\n")
                self.proc.stdin.flush()
            except Exception as e:
                raise RuntimeError(f"[{self.adapter_name}] Failed to write to runner stdin: {e}")

            if not event.wait(timeout=self.request_timeout):
                raise TimeoutError(f"[{self.adapter_name}] Request {rid} timed out after {self.request_timeout}s")

            response = self._futures[rid]["result"]
            if not response:
                raise RuntimeError(f"[{self.adapter_name}] Runner returned no response")

            if not response.get("ok"):
                raise RuntimeError(f"Runner Error: {response.get('error')}")

            return response.get("result")
        finally:
            with self._lock:
                self._futures.pop(rid, None)

    def run_all(self, init, load_model, clone, text, adapter_args):
        with tempfile.TemporaryDirectory(prefix=f"tts_{self.adapter_name}_") as tmp_dir_str:
            tmp_path = Path(tmp_dir_str)

            safe_clone = self._stage_files(clone, tmp_path)
            safe_args = self._stage_files(adapter_args, tmp_path)

            return self._req("run", {
                "init": init,
                "load_model": load_model,
                "clone_voice": safe_clone,
                "synthesize": {"text": text, "kwargs": safe_args},
            })


class RunnerManager:
    def __init__(self):
        self._runners: Dict[str, RunnerProc] = {}
        self._lock = threading.Lock()

    def get(self, adapter_name: str) -> RunnerProc:
        if adapter_name not in ADAPTER_REGISTRY:
            raise ValueError(f"Unknown adapter: {adapter_name}")

        with self._lock:
            # NOTE: design choice: only one runner alive at a time to save VRAM.
            for name, runner in self._runners.items():
                if name != adapter_name and runner.proc is not None and runner.proc.poll() is None:
                    logger.info(f"[Manager] Switching models: Closing {name} to start {adapter_name}")
                    runner.stop()

            if adapter_name not in self._runners:
                self._runners[adapter_name] = RunnerProc(adapter_name)

            return self._runners[adapter_name]


manager = RunnerManager()