import os
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from celery.result import AsyncResult
from tts_playground.tasks.celery_app import app as celery_app


app = FastAPI(title="TTS Playground API", version="0.0.1")


class TTSPayload(BaseModel):
    adapter: str
    init: Dict[str, Any] = {}
    load_model: Dict[str, Any] = {}
    clone_voice: Dict[str, Any] = {}
    synthesize: Dict[str, Any]


class TaskStatus(BaseModel):
    state: str
    sr: int = None
    wav_b64: str = None
    error: str = None


@app.post("/v1/tts", response_model=Dict[str, str])
async def create_tts_task(payload: TTSPayload):
    task = celery_app.send_task("tts.all", kwargs={"payload": payload.model_dump()})
    return {"task_id": task.id}


@app.get("/v1/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    if res.state == "PENDING":
        response = {"state": "PENDING"}
    elif res.state != "FAILURE":
        result = res.result
        response = {"state": "SUCCESS", "sr": result["sr"], "wav_b64": result["wav_b64"]}
    else:
        response = {"state": "FAILURE", "error": str(res.result)}
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "tts_playground.api.main:app",
        host=os.getenv("UVICORN_HOST", "0.0.0.0"),
        port=int(os.getenv("UVICORN_PORT", 7000)),
        log_level="info",
    )