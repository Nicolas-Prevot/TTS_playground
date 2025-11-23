from typing import Any, Dict

from tts_playground.tasks.celery_app import app
from tts_playground.runtime.runner_manager import manager


@app.task(name="tts.all")
def task_all(payload: Dict[str, Any]) -> Dict[str, Any]:
    adapter = payload["adapter"]
    init = payload.get("init", {})
    load_model = payload.get("load_model", {})
    clone = payload.get("clone_voice", {})
    text = payload["synthesize"]["text"]
    adapter_args = payload["synthesize"].get("kwargs", {})
    rp = manager.get(adapter)
    result = rp.run_all(init, load_model, clone, text, adapter_args)
    return result