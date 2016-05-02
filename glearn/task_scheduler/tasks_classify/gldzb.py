# -*- coding: utf-8 -*-
import os
import sys
import commands

from celery import Celery
from kombu import Exchange, Queue

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../../..'))
from glearn.classify.classify_predict_platform import AsyncJobWorker
from glearn.task_scheduler.celery_config import Config, Settings



IDENTITY_CURRENT = os.path.basename(__file__).rstrip('.py')
Config.CELERY_QUEUES.append(
	Queue(IDENTITY_CURRENT, Exchange(Settings.EXCHANGE), routing_key = Settings.EXCHANGE+'.'+IDENTITY_CURRENT)
)
app = Celery()
app.config_from_object(Config)

@app.task
def predict(identity, docs):
	return AsyncJobWorker.classify(identity, docs)


if __name__ == '__main__':
	pass
	# Usage: gldjc.py <command> [options]
	# Show help screen and exit.
	# Options:
	#   -A APP, --app=APP     app instance to use (e.g. module.attr_name)
	#   -b BROKER, --broker=BROKER
	#                         url to broker.  default is 'amqp://guest@localhost//'
	#   --loader=LOADER       name of custom loader class to use.
	#   --config=CONFIG       Name of the configuration module
	#   --workdir=WORKING_DIRECTORY
	#                         Optional directory to change to after detaching.
	#   -C, --no-color        
	#   -q, --quiet           
	#   --version             show program's version number and exit
	#   -h, --help            show this help message and exit

	# ---- -- - - ---- Commands- -------------- --- ------------
	# app.start()