#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../..'))

import requests
import ujson
import unittest
import datetime

import jieba
class TestApiFunctions(unittest.TestCase):
    
    #@unittest.skip("skip")
    def test_api_ClassifyPredictHandler(self):
        print "test_api_ClassifyPredictHandler"

        data = {
                "identity":"gldjc",
                "docs":
                    "1 III级螺纹钢   Φ18mm  HRB400 品种:螺纹钢筋;牌号:HRB400;直径Φ(mm):18\n"+#0101
                    "2 三级螺纹钢   16 直径Φ(mm):16;品种:螺纹钢筋;牌号:HRB400\n"+#0103
                    "3 3809耐油工业橡胶板   厚×宽：1~80mm×500~3000mm 厚度δ（）:1~80;品种:耐油橡胶板\n"+#0201
                    "4 NY150耐油石棉橡胶板   厚×宽：1~80mm×500~3000mm 厚度δ（）:1~80;品种:石棉橡胶板\n"+#0201
                    "5 DS型钢弹簧减振器   KL-19  最佳荷载：170kg \n"+#0203
                    "6 \n"+ #0203
                    "7 内丝组合活接 DN25×3/4 个 尚治 PB"
        }
        print data['docs']

        json_result = requests.post('http://localhost:9701/classify/predict', params=data)
        print ujson.loads(json_result.content)
        


# Run test cases.
if __name__ == '__main__':
    unittest.main()




