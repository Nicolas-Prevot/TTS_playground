import time
import base64
from pathlib import Path
from typing import Any, Dict, Optional, Union
import httpx


class TTSClient:
    def __init__(self, base_url: str = "http://localhost:7000", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def _pack_file(self, fpath: Union[str, Path]) -> Dict[str, str]:
        p = Path(fpath)
        if not p.exists():
            raise FileNotFoundError(f"{p} does not exist")
        return {
            "name": p.name,
            "b64": base64.b64encode(p.read_bytes()).decode("ascii")
        }

    def _pack_recursive(self, data: Any) -> Any:
        """Helper to look for file paths in params and convert to B64 blobs."""
        # This logic allows the user to pass strings in the kwargs, 
        # and if they look like files, we pack them. 
        # Note: This is a bit magic. In the previous code, the user explicitly called pack_file.
        # We will keep the explicit method for safety.
        return data

    def pack_file(self, fpath: str) -> Dict[str, str]:
        return self._pack_file(fpath)

    def pack_speaker_map(self, mapping: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        return {k: self._pack_file(v) for k, v in mapping.items()}

    def synth(
        self,
        adapter: str,
        init: Optional[Dict] = None,
        load_model: Optional[Dict] = None,
        clone_voice: Optional[Dict] = None,
        synthesize: Optional[Dict] = None,
        wait: bool = True,
        dest_path: Optional[str] = None,
        timeout: Optional[float] = None,
        download: bool = True,
    ) -> Dict[str, Any]:
        
        payload = {
            "adapter": adapter,
            "init": init or {},
            "load_model": load_model or {},
            "clone_voice": clone_voice or {},
            "synthesize": synthesize or {"text": "Hello", "kwargs": {}}
        }

        r = self.client.post("/v1/tts", json=payload)
        r.raise_for_status()
        task_id = r.json()["task_id"]

        if not wait:
            return {"state": "SUBMITTED", "task_id": task_id}

        # Polling
        to = timeout or self.timeout
        start = time.time()
        while (time.time() - start) < to:
            r = self.client.get(f"/v1/tasks/{task_id}")
            if r.status_code == 200:
                data = r.json()
                state = data["state"]
                if state in ["SUCCESS", "FAILURE"]:
                    if state == "SUCCESS" and download and dest_path and data.get("wav_b64"):
                        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
                        Path(dest_path).write_bytes(base64.b64decode(data["wav_b64"]))
                        del data["wav_b64"] # Cleanup memory
                    return data
            time.sleep(1)
        
        raise TimeoutError(f"Task {task_id} timed out")

    def close(self):
        self.client.close()