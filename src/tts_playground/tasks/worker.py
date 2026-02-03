from typing import Any, Dict
import gc
from tts_playground.tasks.celery_app import celery_app
from tts_playground.runtime.runner_manager import manager


@celery_app.task(name="tts.all", bind=True)
def task_all(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main worker entry point.
    """
    adapter_name = payload["adapter"]
    if not adapter_name:
        raise ValueError("payload.adapter is required")
    
    init_params = payload.get("init", {})
    load_params = payload.get("load_model", {})
    clone_params = payload.get("clone_voice", {})
    synth_data = payload["synthesize"]
    
    text = synth_data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("payload.synthesize.text must be a non-empty string")

    synth_kwargs = synth_data.get("kwargs") or {}
    if not isinstance(synth_kwargs, dict):
        raise ValueError("payload.synthesize.kwargs must be an object/dict")

    runner = manager.get(adapter_name)
    
    result = runner.run_all(
        init=init_params,
        load_model=load_params,
        clone=clone_params,
        text=text,
        adapter_args=synth_kwargs
    )
    return result

@celery_app.task(name="tts.stop_runners", bind=True)
def task_stop_runners(self) -> dict:
    """
    Stops all adapter runner subprocesses to free memory immediately.
    """
    stopped = manager.stop_all()

    gc.collect()
    try:
        import torch  # optional; worker env may not have it
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    return {"stopped": stopped}