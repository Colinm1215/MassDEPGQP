from celery import Celery
import redis

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

celery = Celery(
    'MassDEP',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery.conf.update(
    result_expires=3600,
    task_serializer='pickle',
    accept_content=['pickle', 'json'],
    result_serializer='pickle',
    timezone='UTC',
    enable_utc=True,
)

celery.conf.update(
    broker_heartbeat=0,
    broker_transport_options={
        'visibility_timeout': 3600,
        'socket_keepalive': True,
        'tcp_keepalive': True,
    }
)
