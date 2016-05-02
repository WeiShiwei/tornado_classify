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

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../..'))
from glearn.classify import orm
from glearn.classify import global_variables as gv
from glearn.feature_engineering import Feature_Engineering

from preprocessing_util import PreprocessingUtil
import refine_options 

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../'))
import shared_variables as sv

reload(sys)                         #  
sys.setdefaultencoding('utf-8')     # 


class GoogleRefine(object):
	"""docstring for CustomTabularExporter"""
	def __init__(self, corpus_path=refine_options.LEARN_CORPUS_PATH):
		super(GoogleRefine, self).__init__()
		self.corpus_path = corpus_path
		self.CustomTabularExporter_OPTIONS = refine_options.CustomTabularExporter_OPTIONS

	# @classmethod
	def custom_tabular_export(self, project_id, lv2_code):
		""" custom_tabular_export """
		self.CustomTabularExporter_OPTIONS['project'] = project_id
		url = ''.join( [refine_options.EXPORT_URL, lv2_code, '.tsv'] )
		
		error_of_attempts = 5
		for i in range(error_of_attempts):
			try:
				json_result = requests.post(url, params=self.CustomTabularExporter_OPTIONS)
				break
			except Exception, e:
				print e
				time.sleep(5)
				continue
		
		return json_result

	# @classmethod
	def get_all_project_metadata(self):
		json_result = requests.get(refine_options.GET_ALL_PROJECT_METADATA_URL, params={})
		project_metadata = ujson.loads(json_result.content)
		return project_metadata

	# @classmethod
	def _write_corpus(self, first_type_code, second_type_code, refine_file_path):
		# import pdb;pdb.set_trace()
		
		# 如果是更新测试语料的话,工作已经完成
		if refine_file_path.startswith( refine_options.TEST_CORPUS_PATH ):
			return
		train_contents_fe = PreprocessingUtil.preprocess_source_file(refine_file_path)
		outfile_path = os.path.join(self.corpus_path, 
									first_type_code, 
									second_type_code, 
									first_type_code+'.'+second_type_code+'.train')
		print outfile_path
		fout = open(outfile_path ,'w')
		fout.write('\n'.join( sorted(train_contents_fe)) )
		fout.close()

	# @classmethod
	def update_classify_data(self, lv2_code, json_result_content):
		""""""
		first_type_code = lv2_code[:2]
		second_type_code = lv2_code
		target_directory = os.path.join( self.corpus_path, first_type_code, second_type_code)
		if not os.path.exists( target_directory ):
			os.mkdir( target_directory )
		
		# 拉取google-refine.gldjc.com的清洗后的数据写入tsv文件
		refine_file_path = os.path.join(target_directory, '.'.join([first_type_code,second_type_code,'refine','tsv']))
		print refine_file_path
		with open( refine_file_path, 'wb') as infile:
			infile.write( json_result_content )

		# ----------------------------------------
		return refine_file_path

	def load_metadata(self):
		# 考虑不周到的地方，如果备份的数据有多于google-refine的类别呢
		# 只许进，不许删
		project_metadata_backup_dict = dict()
		try:
			with open('./project_metadata.json','r') as f:
				project_metadata_backup = ujson.loads( f.read())
				for project_id,meta in project_metadata_backup['projects'].items():
					lv2_code = meta['name']
					project_metadata_backup_dict[lv2_code] = meta['modified']
		except Exception, e:
			print 'ERROR:load project_metadata.json fails! so asscess google-refine for project_metadata'
		return project_metadata_backup_dict
	
	def dump_metadata(self, project_metadata):
		# 备份
		with open('./project_metadata.json','w') as f:
			f.write( ujson.dumps(project_metadata) )

	# @classmethod
	def preprocess(self, multilevel_code):
		""" multilevel_code应该是个空字符串或者lv1_code
		python preprocess 01
		python preprocess 
		"""
		project_metadata = self.get_all_project_metadata()
		project_metadata_backup_dict = self.load_metadata()
		
		projects = list()
		for project_id,meta in project_metadata['projects'].items():
			lv2_code = meta['name'];lv1_code = lv2_code[:2]
			modified_time = meta['modified']
			
			if (multilevel_code == '') or (lv1_code == multilevel_code):
				# print lv2_code
				projects.append( (project_id,lv2_code,modified_time) )

		for project_id,lv2_code,modified_time in projects:
			lv1_code = lv2_code[:2]

			if modified_time == project_metadata_backup_dict.get(lv2_code, '0000-00-00T00:00:00Z'):
				print lv2_code+">>>"
				# import pdb;pdb.set_trace() ###
				refine_file_path = os.path.join(
					os.path.join( self.corpus_path, lv1_code, lv2_code), 
					'.'.join([lv1_code,lv2_code,'refine','tsv'])
				)
				self._write_corpus(lv1_code, lv2_code, refine_file_path)
				continue
			print lv2_code+">>>export>>>"
			json_result = self.custom_tabular_export(project_id, lv2_code)
			refine_file_path = self.update_classify_data(lv2_code, json_result.content)
			self._write_corpus(lv1_code, lv2_code, refine_file_path)

		# self.dump_metadata(project_metadata)


	def test_preprocess(self, filtered_prefix):
		project_metadata = self.get_all_project_metadata()
		# project_metadata_backup_dict = self.load_metadata()
		
		projects = list()
		for project_id,meta in project_metadata['projects'].items():
			project_name = meta['name']
			modified_time = meta['modified']
			
			if filtered_prefix == '' or project_name.startswith(filtered_prefix):
				projects.append( (project_id, project_name, modified_time) )

		for project_id,project_name,modified_time in projects:
			print project_name+">>>export>>>"
			json_result = self.custom_tabular_export(project_id, project_name)
			# refine_file_path = self.update_classify_data(project_name, json_result.content)
			with open(os.path.join(refine_options.TEST_CORPUS_PATH, project_id+'#'+project_name+'.tsv'), 'wb') as outfile:
				outfile.write(json_result.content)


def main():
	print """usage: python preprocessing 01 | python preprocessing 
					python preprocessing test """

	try:
		argv1 = sys.argv[1]
	except Exception, e:
		argv1 = ''
	print sys.argv

	if (argv1 in sv.all_lv1_codes) or (argv1 == ''):
		multilevel_code = argv1
		gr = GoogleRefine()
		gr.preprocess( multilevel_code )
	elif argv1=='test':
		gr = GoogleRefine(corpus_path = refine_options.TEST_CORPUS_PATH)
		gr.test_preprocess( 'TEST' )
	else:
		print 'error command line'





if __name__ == '__main__':
	main()