from typing import Any, Dict

from tts_playground.tasks.celery_app import celery_app
from tts_playground.runtime.runner_manager import manager


@celery_app.task(name="tts.all", bind=True)
def task_all(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main worker entry point.
    """
    adapter_name = payload["adapter"]
    
    init_params = payload.get("init", {})
    load_params = payload.get("load_model", {})
    clone_params = payload.get("clone_voice", {})
    synth_data = payload["synthesize"]
    
    text = synth_data["text"]
    synth_kwargs = synth_data.get("kwargs", {})

    runner = manager.get(adapter_name)
    
    result = runner.run_all(
        init=init_params,
        load_model=load_params,
        clone=clone_params,
        text=text,
        adapter_args=synth_kwargs
    )
    return result