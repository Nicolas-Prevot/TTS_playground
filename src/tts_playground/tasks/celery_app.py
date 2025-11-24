import os
from celery import Celery
from kombu import Queue

BROKER = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
BACKEND = os.getenv("CELERY_RESULT_BACKEND", BROKER)

celery_app = Celery(
    "tts_worker",
    broker=BROKER,
    backend=BACKEND,
    include=["tts_playground.tasks.worker"]
)

celery_app.conf.task_queues = (Queue("tts.dynamic"),)
celery_app.conf.task_default_queue = "tts.dynamic"
celery_app.conf.task_default_exchange = "tts"
celery_app.conf.result_expires = 3600
celery_app.conf.accept_content = ['json']
celery_app.conf.task_serializer = 'json'
celery_app.conf.result_serializer = 'json'