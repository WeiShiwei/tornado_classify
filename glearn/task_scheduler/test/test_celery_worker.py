# -*- coding: utf-8 -*-
import os, sys

import requests
import ujson
import unittest
import datetime
import time

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../../..'))
from glearn.task_scheduler.tasks_classify import gldjc, gldzb

class TestApiFunctions(unittest.TestCase):
	
	@unittest.skip("skip")
	def test_api_gldjc_celery_worker(self):
		print 'test_api_gldjc_celery_worker'
		# weishiwei@hadoop1:~/800w_classifier/tornado_classify/glearn/task_scheduler$ \
		# celery worker -E -l INFO -n workerA -Q gldjc -A tasks_classify.gldjc  --concurrency=1
		# 命令行中-Q参数确定了queue和worker的绑定
		identity = 'gldjc'
		docs =['III级螺纹钢   Φ18mm  HRB400 品种:螺纹钢筋;牌号:HRB400;直径Φ(mm):18','3809耐油工业橡胶板   厚×宽：1~80mm×500~3000mm 厚度δ（）:1~80;品种:耐油橡胶板','DS型钢弹簧减振器   KL-19  最佳荷载：170kg']
		modellearning = 'hierarchical'

		res = gldjc.predict.apply_async((identity,docs,modellearning),queue=identity)
		print gldjc.predict
		print res
		print res.status
		time.sleep(3)
		print res.result

	@unittest.skip("skip")
	def test_api_gldzb_celery_worker(self):
		print 'test_api_gldzb_celery_worker'
		identity = 'gldzb'
		docs =['III级螺纹钢   Φ18mm  HRB400 品种:螺纹钢筋;牌号:HRB400;直径Φ(mm):18','3809耐油工业橡胶板   厚×宽：1~80mm×500~3000mm 厚度δ（）:1~80;品种:耐油橡胶板','DS型钢弹簧减振器   KL-19  最佳荷载：170kg']
		modellearning = 'hierarchical'

		res = gldzb.predict.apply_async((identity,docs,modellearning),queue=identity)
		print gldzb.predict
		print res.result
		print res
		print res.status
		# time.sleep(3)
		print res.result
		import pdb;pdb.set_trace()

	#@unittest.skip("skip")
	def test_api_gldzb_celery_worker_learn(self):
		identity = 'gldzb'
		archive_file_name = '20news-bydate-part.tar.gz'
		model_learning_type = 'hierarchical'

		res = gldzb.learn.apply_async((identity,archive_file_name,model_learning_type),queue=identity)
		print gldzb.predict
		print res.result
		print res
		print res.status
		# time.sleep(3)
		print res.result
		import pdb;pdb.set_trace()



# Run test cases.
if __name__ == '__main__':
    unittest.main()
