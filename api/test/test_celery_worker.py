#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../..'))

import requests
import ujson
import unittest
import datetime
import time

import jieba
class TestApiFunctions(unittest.TestCase):

    #@unittest.skip("skip")
    def test_api_ApplyAsyncHandler(self):
        print "test_api_ApplyAsyncHandler"

        data = {
                "identity":"gldzb",
                'docs': ujson.dumps([
                    'III级螺纹钢   Φ18mm  HRB400 品种:螺纹钢筋;牌号:HRB400;直径Φ(mm):18',
                    '3809耐油工业橡胶板   厚×宽：1~80mm×500~3000mm 厚度δ（）:1~80;品种:耐油橡胶板',
                    'DS型钢弹簧减振器   KL-19  最佳荷载：170kg',
                ])
        }
        json_result = requests.post('http://127.0.0.1:9701/tasks/apply-async', params=data)
        print ujson.loads(json_result.content)

        # import pdb;pdb.set_trace()
        task_id = ujson.loads(json_result.content)["task_id"]
        time.sleep(5)
        data = {
                "task_id":task_id
        }
        json_result = requests.post('http://127.0.0.1:9701/tasks/result', params=data)
        print ujson.loads(json_result.content)        

    @unittest.skip("skip")
    def test_api_CeleryResultHandler(self):
        data = {
            # celery-task-meta-
            "task_id":'e6905c08-b64c-413f-8bd7-f019457bbf68'
        }
        json_result = requests.post('http://127.0.0.1:9701/tasks/result', params=data)
        print ujson.loads(json_result.content)


# Run test cases.
if __name__ == '__main__':
    unittest.main()




