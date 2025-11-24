import os
import subprocess
import threading
import uuid
import json
import sys
import base64
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union
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
}

ADAPTERS_ROOT = Path(os.getenv("RUNNER_VENVS_DIR", "./adapters")).resolve()

class RunnerProc:
    def __init__(self, adapter_name: str):
        self.adapter_name = adapter_name
        self.config = ADAPTER_REGISTRY[adapter_name]
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._futures: Dict[str, Any] = {}
        
        self.idle_secs = int(os.getenv("IDLE_SECS", "180"))
        self.exit_on_idle = os.getenv("EXIT_ON_IDLE", "1") == "1"

    def _start_process(self):
        folder_name, mod_path, cls_name = self.config
        adapter_dir = ADAPTERS_ROOT / folder_name
        
        if sys.platform == "win32":
            python_exe = adapter_dir / ".venv" / "Scripts" / "python.exe"
        else:
            python_exe = adapter_dir / ".venv" / "bin" / "python"

        if not python_exe.exists():
            raise RuntimeError(f"Venv not found at {python_exe}. Run 'uv sync' in {adapter_dir}")

        runner_script = Path(__file__).parent / "adapter_runner.py"
        
        cmd = [
            str(python_exe),
            str(runner_script),
            "--module", mod_path,
            "--cls", cls_name,
            "--idle", str(self.idle_secs),
            "--exit-on-idle", "1" if self.exit_on_idle else "0"
        ]

        logger.info(f"[{self.adapter_name}] Spawning: {' '.join(cmd)}")
        
        # Create a clean environment but pass necessary variables
        env = os.environ.copy()
        
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, 
            text=True,
            bufsize=1,
            env=env,
            cwd=str(adapter_dir)
        )
        
        t_out = threading.Thread(target=self._read_loop, daemon=True)
        t_out.start()

        t_err = threading.Thread(target=self._stderr_loop, daemon=True)
        t_err.start()

    def _read_loop(self):
        """Reads JSON responses from STDOUT."""
        while self.proc and self.proc.poll() is None:
            try:
                line = self.proc.stdout.readline()
                if not line: break
                line = line.strip()
                if not line: continue

                msg = json.loads(line)
                rid = msg.get("id")
                
                with self._lock:
                    if rid in self._futures:
                        self._futures[rid]['result'] = msg
                        self._futures[rid]['event'].set()
            except json.JSONDecodeError:
                # Fallback if non-json creeps into stdout
                logger.debug(f"[{self.adapter_name} RAW] {line}")
            except Exception as e:
                logger.error(f"[{self.adapter_name}] Read loop error: {e}")

    def _stderr_loop(self):
        """Reads logs from STDERR and pipes them to the main logger."""
        while self.proc and self.proc.poll() is None:
            try:
                line = self.proc.stderr.readline()
                if not line: break
                line = line.strip()
                if line:
                    # Log as INFO so it shows up in Celery logs
                    logger.info(f"[{self.adapter_name}] {line}")
            except Exception:
                break

    def _is_blob(self, item: Any) -> bool:
        return isinstance(item, dict) and "b64" in item and "name" in item

    def _stage_files(self, data: Any, temp_dir: Path) -> Any:
        """Recursively find B64 blobs, write to disk, replace with Path str."""
        if isinstance(data, dict):
            if self._is_blob(data):
                # It's a file blob -> Write it
                safe_name = "".join(c for c in data["name"] if c.isalnum() or c in "._-")
                file_path = temp_dir / safe_name
                file_path.write_bytes(base64.b64decode(data["b64"]))
                return str(file_path)
            else:
                return {k: self._stage_files(v, temp_dir) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._stage_files(i, temp_dir) for i in data]
        else:
            return data

    def _req(self, method: str, params: Dict[str, Any]) -> Any:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start_process()

        rid = uuid.uuid4().hex
        event = threading.Event()
        
        with self._lock:
            self._futures[rid] = {'event': event, 'result': None}

        try:
            payload = {"id": rid, "method": method, "params": params}
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
            
            if not event.wait(timeout=600):
                raise TimeoutError(f"Request {rid} timed out")
            
            response = self._futures[rid]['result']
            if not response.get("ok"):
                raise RuntimeError(f"Runner Error: {response.get('error')}")
            
            return response.get("result")
        finally:
            with self._lock:
                self._futures.pop(rid, None)

    def run_all(self, init, load_model, clone, text, adapter_args):
        # Create a temporary directory for this specific request
        with tempfile.TemporaryDirectory(prefix=f"tts_{self.adapter_name}_") as tmp_dir_str:
            tmp_path = Path(tmp_dir_str)
            
            # 1. Stage files (convert B64 -> Paths)
            safe_clone = self._stage_files(clone, tmp_path)
            safe_args = self._stage_files(adapter_args, tmp_path)
            
            # 2. Call Runner
            result = self._req("run", {
                "init": init,
                "load_model": load_model,
                "clone_voice": safe_clone,
                "synthesize": {"text": text, "kwargs": safe_args}
            })
            
            # 3. Temp dir is automatically cleaned up here
            return result

class RunnerManager:
    def __init__(self):
        self._runners: Dict[str, RunnerProc] = {}
        self._lock = threading.Lock()

    def get(self, adapter_name: str) -> RunnerProc:
        if adapter_name not in ADAPTER_REGISTRY:
            raise ValueError(f"Unknown adapter: {adapter_name}")
        with self._lock:
            if adapter_name not in self._runners:
                self._runners[adapter_name] = RunnerProc(adapter_name)
            return self._runners[adapter_name]

manager = RunnerManager()