from celery import Celery
import redis

# Initializes Redis client and configures Celery application for asynchronous task processing.

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

celery = Celery(
    'MassDEP',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery.conf.update(
    result_expires=3600,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    broker_heartbeat=0,
    broker_transport_options={
        'visibility_timeout': 3600,
        'socket_keepalive': True,
        'tcp_keepalive': True,
    }
)
