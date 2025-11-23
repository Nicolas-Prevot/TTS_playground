import os
from celery import Celery
from kombu import Queue

BROKER = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL") or "redis://redis:6379/0"
BACKEND = os.getenv("CELERY_RESULT_BACKEND") or BROKER

app = Celery("tts", broker=BROKER, backend=BACKEND, include=["tts_playground.tasks.tts_tasks"])

app.conf.task_queues = (Queue("tts.dynamic"),)
app.conf.task_default_queue = "tts.dynamic"
app.conf.task_default_exchange = "tts"
app.conf.result_expires = 3600