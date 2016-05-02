#!/usr/bin/python
# coding:utf-8

import os,sys
import collections
import csv
import requests
import ujson
import fnmatch
import xlrd
import time
import glob
import xlsxwriter

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../..'))
from glearn.datasets import load_files as load_files_bunch
from glearn.datasets.load_files_platform import load_files # 训练一级类模型是有问题的
from glearn.classify import global_variables as gv
from glearn.classify.classify_predict import ClassifyEngine

from xlsx_writer import XlsxWriter
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../'))
import shared_variables as sv


reload(sys)                         #  
sys.setdefaultencoding('utf-8')     # 




TEST_CORPUS_PATH = os.path.expanduser( os.path.join( '~','scikit_test_data' ) )
REFINE_CORPUS_PATH = os.path.join(TEST_CORPUS_PATH, 'Google-refine')
CODE_FIELD_NAME = u'类别编码'
POSTFIX = 'train'




class ClassifyTest(object):
	"""docstring for ClassifyTest"""
	code_documents_dict = collections.defaultdict(list)
	
	@classmethod
	def update_test_corpus(self):
		print REFINE_CORPUS_PATH
		
		corpus_files = glob.glob( os.path.join(REFINE_CORPUS_PATH, '*.tsv') )
		for cf in corpus_files:
			reader = csv.reader(file(cf, 'rb'), delimiter='\t')
			header = True
			for line in reader:
				if header:
					try:
						# import pdb;pdb.set_trace()
						code_field_idx = line.index(CODE_FIELD_NAME)
					except Exception, e:
						code_field_idx = -1
						print 'error: no CODE_FIELD'
					header = False
					continue
				if not line:
					continue
				try:
					lv2_code = line[code_field_idx]
					int(lv2_code)
					del line[code_field_idx]
					self.code_documents_dict[lv2_code].append( ' '.join(line) )
				except Exception, e:
					print line

		for lv2_code,documents in self.code_documents_dict.items():
			lv1_code = lv2_code[:2]
			target_dir = os.path.join(TEST_CORPUS_PATH,lv1_code,lv2_code)
			if not os.path.exists(target_dir):
				os.makedirs(target_dir)
			with open( os.path.join(target_dir, '.'.join( [lv1_code, lv2_code, POSTFIX]) ), 'wb') as outfile:
				outfile.write( '\n'.join(documents) )

	@classmethod
	def test(self, multilevel_code, categories_select=None):
		gv.CORPUS_PATH = TEST_CORPUS_PATH # 转换语料库

		data_test = load_files_bunch.fetch_Bunch_datas(multilevel_code, categories=categories_select)
		json_result = ClassifyEngine.classify_test(data_test)

		# print "===整体的评估情况"
		# print "accuracy_score: ",json_result["accuracy_score"]
		# print "recall_score: ",json_result["recall_score"]
		# print "f1_score: ",json_result["f1_score"]
		# print "classification_report: \n",json_result["classification_report"]
		# print "confusion_matrix: \n",json_result["confusion_matrix"]
		# print "===文本分类系统预测出错的文档列表"
		
		# for document in json_result["error_predict_documents"]:
		# 	print "="*40
		# 	print "document: ",document["document"].strip()
		# 	print "multi_patterns: ",document["multi_patterns"]
		# 	print "ytest: ",document["ytest"]
		# 	print "ypred: ",document["ypred"]

		# return json_result
		# -------------------------------------------
		
		filepath = os.path.join(os.path.expanduser( os.path.join( '~','Documents' ) ), 'classification_report.xlsx')
		print filepath
		xw = XlsxWriter(filepath)
		xw.write( json_result )
		



def main():
	"""
	独立测试用于评估模型
	python classify-testing 01 | python classify-testing 
	
	把scikit_test_data/Google-refine目录下的内容更新到scikit_test_data
	python classify-testing update
	"""
	print """usage: python classify-testing 01 | python classify-testing  | python classify-testing update"""
	
	try:
		argv1 = sys.argv[1]
	except Exception, e:
		argv1 = ''
	
	print sys.argv
	print "argv1:",argv1

	if argv1 == 'update':
		ClassifyTest.update_test_corpus()
		sys.exit()
	elif argv1 in sv.all_lv1_codes:
		multilevel_code = argv1
		json_result = ClassifyTest.test( multilevel_code, categories_select=None )
	else:
		print 'error commang line'
		


if __name__ == '__main__':
	main()