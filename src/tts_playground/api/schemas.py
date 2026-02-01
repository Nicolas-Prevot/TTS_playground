from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class SynthesizePayload(BaseModel):
    text: str
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class TTSPayload(BaseModel):
    adapter: str
    init: Dict[str, Any] = Field(default_factory=dict)
    load_model: Dict[str, Any] = Field(default_factory=dict)
    clone_voice: Dict[str, Any] = Field(default_factory=dict)
    synthesize: SynthesizePayload


class TaskStatus(BaseModel):
    state: str
    task_id: str
    sr: Optional[int] = None
    wav_b64: Optional[str] = None
    error: Optional[str] = None