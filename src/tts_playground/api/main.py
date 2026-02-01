import os

from fastapi import FastAPI, HTTPException
from celery.result import AsyncResult

from tts_playground.api.schemas import TTSPayload, TaskStatus
from tts_playground.tasks.celery_app import celery_app
from tts_playground.runtime.runner_manager import ADAPTER_REGISTRY


app = FastAPI(title="TTS Playground API", version="0.0.1")


@app.post("/v1/tts", response_model=dict[str, str])
async def create_tts_task(payload: TTSPayload):
    if payload.adapter not in ADAPTER_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown adapter '{payload.adapter}'. "
                   f"Available: {', '.join(sorted(ADAPTER_REGISTRY.keys()))}"
        )

    task = celery_app.send_task("tts.all", kwargs={"payload": payload.model_dump()})
    return {"task_id": task.id}


@app.get("/v1/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    res = AsyncResult(task_id, app=celery_app)

    response = TaskStatus(state=res.state, task_id=task_id)

    if res.state == "SUCCESS":
        result = res.result or {}
        response.sr = result.get("sr")
        response.wav_b64 = result.get("wav_b64")
    elif res.state == "FAILURE":
        response.error = str(res.result)

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "tts_playground.api.main:app",
        host=os.getenv("UVICORN_HOST", "0.0.0.0"),
        port=int(os.getenv("UVICORN_PORT", 7000)),
        log_level="info",
    )
