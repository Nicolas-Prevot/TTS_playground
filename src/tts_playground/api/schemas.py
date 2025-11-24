from typing import Any, Dict, Optional
from pydantic import BaseModel

class TTSPayload(BaseModel):
    adapter: str
    init: Dict[str, Any] = {}
    load_model: Dict[str, Any] = {}
    clone_voice: Dict[str, Any] = {}
    synthesize: Dict[str, Any]

class TaskStatus(BaseModel):
    state: str
    task_id: str
    sr: Optional[int] = None
    wav_b64: Optional[str] = None
    error: Optional[str] = None