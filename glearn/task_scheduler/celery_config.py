# -*- coding: utf-8 -*-
# from __future__ import absolute_import

from celery import Celery
from kombu import Exchange, Queue


class Config:
	BROKER_URL = 'amqp://guest:guest@localhost:5672//'
	CELERY_RESULT_BACKEND = 'redis://localhost/0'

	CELERY_TASK_SERIALIZER = 'json'
	CELERY_RESULT_SERIALIZER = 'json'
	CELERY_ACCEPT_CONTENT=['json']
	# CELERY_TIMEZONE = 'Europe/Oslo'
	CELERY_ENABLE_UTC = True

	# Optional configuration, see the application user guide.
	CELERY_TASK_RESULT_EXPIRES=3600

	
	# such as:(Queue('gldjc', Exchange(Settings.EXCHANGE), routing_key = 'classify.gldjc'), ...)
	CELERY_QUEUES = list()
	# 命令行中指定队列gldjc和（相关多）任务tasks_classify.gldjc的绑定
	# such as:{'tasks_classify.gldjc.predict': {'queue': 'gldjc', 'routing_key':'classify.gldjc'}, ...}
	CELERY_ROUTES = list()


class Settings:
	EXCHANGE = 'classify'