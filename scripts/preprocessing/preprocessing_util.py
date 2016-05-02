#!/usr/bin/python
# coding:utf-8

import os,sys
import collections
import csv
import requests
import ujson
import fnmatch
import xlrd

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../..'))
from glearn.feature_engineering import Feature_Engineering as FeatureEngineering

ESCAPE_LV2_CODES = ['0001',
					'0127','0163',
					'0207','0219','0235',
					'0367',
					'0525',
					'0643','0679',
					'0711','0755',
					'0819','0837',
					'0927',
					'1017','1045',
					'1131','1171',
					'1209',
					'1319','1343','1359',
					'1457',
					'1513',
					'1641',
					'1711','1735',
					'1831','1883',
					'1909','1933','1961',
					'2041','2045',
					'2169',
					'2261',
					'2323','2349',
					'2425',
					'2543',
					'2627',
					'2721',
					'2823',
					'2935',
					'3041',
					'3239',
					'3337',
					'5027','5041',
					'5133',
					'5213','5229',
					'5347',
					'5443',# 其他排烟设备
					'5549',
					'5625',
					'5745',
					'5819',
					'8009','8035',
					'9847',
					'9945']
Product = collections.namedtuple("Product","name unit brand spec attrs")



class PreprocessingUtil(object):
	"""docstring for BasePreprocessing"""

	FIELD_FILTER = 'all'
	# FIELD_FILTER = 'name'

	@classmethod
	def process_by_field(self, product_tuple):
		try:
			name = product_tuple.name.strip() if product_tuple.name is not None else ''
			unit = product_tuple.unit.strip()[:0] if product_tuple.unit is not None else ''
			brand =product_tuple.brand.strip()[:0] if product_tuple.brand is not None else ''
			spec = product_tuple.spec.strip() if product_tuple.spec is not None else '' #可以抽取‘型号’、正则表达去掉‘数字’等等，暂时不作处理
			attrs = product_tuple.attrs.strip() if product_tuple.attrs is not None else ''

			content = [name,unit,brand,spec,attrs]
		except Exception, e:
			content = field_values

		return content

	@classmethod
	def _feature_engineering(self, train_contents):
		if PreprocessingUtil.FIELD_FILTER=='name':
			lines = [content[0] for content in train_contents] # 只拿产品的名称作为训练数据
		else:
			lines = [' '.join(content) for content in train_contents]
		# ------------------------------------
		### 去重的策略
		lines_set = set(lines)
		lines = list(lines_set)
		# ------------------------------------
		lines = FeatureEngineering.normalize_lines(lines) ###
		return lines

	@classmethod
	def preprocess_source_file(self, source_file):
		target_dirname = os.path.dirname(source_file)
		templist = os.path.basename( source_file ).split('.')
		first_type_code,second_type_code,source_file_type = [templist[0], templist[1], templist[-1]]

		train_contents = list()
		if source_file_type == 'csv':
			train_contents_fe = list()

			reader = csv.reader(file(source_file, 'rb'), quoting=csv.QUOTE_ALL)
			header = True
			for line in reader:
				if header:
					header = False
					continue
				content = self.process_by_field( Product(*line) )
				train_contents.append(content)
			train_contents_fe = self._feature_engineering(train_contents)
		elif source_file_type == 'tsv':
			train_contents_fe = list()

			reader = csv.reader(file(source_file, 'rb'), delimiter='\t')
			header = True
			for line in reader:
				if header:
					header = False
					continue
				content = self.process_by_field( Product(*line) )
				train_contents.append(content)
			train_contents_fe = self._feature_engineering(train_contents)
		else:
			print "未能处理的文件类型"

		# import pdb;pdb.set_trace()
		return train_contents_fe