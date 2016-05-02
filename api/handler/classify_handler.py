#!/usr/bin/python
# -*- coding: utf-8 -*-

import traceback
from api.handler.base_handler import BaseHandler
import ujson

import sys
import os
import datetime
import collections
from time import time

from glearn.datasets import load_files
from glearn.classify import global_variables as gv
from glearn.classify.classify_predict import ClassifyEngine,Bunch

from lib.log_util import create_log
app_logger = create_log('classify')

import tornado

reload(sys)                         #  
sys.setdefaultencoding('utf-8')     # 
   
class ClassifyIndexHandler(BaseHandler):
	_label = "ClassifyIndexHandler"

	def get(self):
		self.render('index.html')

	def post(self):
		pass
		
class ClassifyPredictHandler(BaseHandler):
	_label = "ClassifyPredictHandler"

	def post(self):
		docs = self.get_argument('docs',default='').split('\n')
		identity = self.get_argument('identity',default='')
		print 'docs:',docs

		ids = list()
		for i in xrange( len(docs) ):
		    doc = docs[i]
		    if doc == '':
		        continue
		    try:
		        sep_pos = doc.index(' ')
		        id = doc[0:sep_pos].strip();int(id)
		        docs[i] = doc[sep_pos+1:].strip()
		    except Exception, e:
		        print 'Error:disqualification doc=>"'+doc+'"'
		        continue
		    ids.append(id)
		# -----------------------------------------
		# multi_patterns_res中的元素不为空即为布尔规则预测的结果
		if not identity:
			classify_res,multi_patterns_res,_,_ = ClassifyEngine.classify(docs) 
		# elif identity=='gldjc': 多层级预测(一般化)
		# 	classify_res = ClassifyEngine.predict(docs) # generalized
		# 	classify_res = map(lambda x:''.join(x.split('.')), classify_res)
		else:
			classify_res = ['NA']*len(ids)
		# -----------------------------------------

		print ids
		print classify_res
		json_result = list()
		for i in xrange(len(ids)):
			json_result.append({'id':ids[i],'category':classify_res[i]})
		try:
			self._json_response(json_result)
		except:
			self.send_error()
			self._app_logger.error(traceback.format_exc())

class ClassifyTestPredictHandler(BaseHandler):
	_label = "ClassifyTestPredictHandler"

	def get(self):
		self.render('test/index.html')

	def post(self):
		try: 
			data_test = Bunch(data=data,
							target_names=target_names,
							target=target,
							DESCR=description)
			json_result = dict()
			self._json_response( json_result )
		except:
			self.send_error()
			self._app_logger.error(traceback.format_exc())

class ClassifyTestUploadFileHandler(BaseHandler):
	_label = "ClassifyTestUploadFileHandler"

	def get(self):
		self.render('test/upload.html')

	def post(self):
		try: 
			testdata_home = gv.get_testdata_home()

			identity = self.get_argument('identity', default='')
			email_address = self.get_argument('email', default='')
			load_files_type = self.get_argument('load_files_type', default='')
			model_learning_type = self.get_argument('model_learning_type', default='')

			file_metas=self.request.files['file']    #提取表单中‘name’为‘file’的文件元数据
			for meta in file_metas:
				file_name=meta['filename']

				identity_home = os.path.join( testdata_home, identity )
				print identity_home
				if not os.path.exists( identity_home ):
					os.makedirs( identity_home )
				file_path =os.path.join( identity_home, file_name ) 
				with open( file_path,'wb' ) as up:   # 有些文件需要已二进制的形式存储，实际中可以更改
					up.write(meta['body'])

			# ============
			# print file_path
			json_result = ClassifyEngine.parse_xlsx_file(file_path)

			self.render('test/result.html', 
				file_name  = file_name,
				accuracy_score=json_result['accuracy_score'], 
				recall_score=json_result['recall_score'], 
				f1_score=json_result['f1_score'], 
				classification_report=json_result['classification_report'],
				confusion_matrix=json_result['confusion_matrix'],
				error_predict_documents=json_result['error_predict_documents'])

		except:
			self.send_error()
			self._app_logger.error(traceback.format_exc())    
	