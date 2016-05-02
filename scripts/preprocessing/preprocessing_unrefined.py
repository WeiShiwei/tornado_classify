#!/usr/bin/python
# coding:utf-8

import os,sys
import collections
import csv
import requests
import ujson
import fnmatch

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../..'))
from glearn.classify import orm
from glearn.classify import global_variables as gv
from glearn.feature_engineering import Feature_Engineering

from preprocessing_util import PreprocessingUtil

reload(sys)                         #  
sys.setdefaultencoding('utf-8')     # 

# corpus_path = gv.CORPUS_PATH
CORPUS_PATH = os.path.expanduser( os.path.join( '~','scikit_learn_data_contrast' ) )
print "CORPUS_PATH:",CORPUS_PATH


def traverse_directory_tree(root, patterns='*.csv', single_level=False, yield_folders=False):
    # multilevelCode_filePath_dict = dict()
    filtered_files = list()
    patterns = patterns.split(';')
    for path, subdirs, files in os.walk(root):
        if yield_folders:
            files.extend(subdirs)
        files.sort()
        for name in files:
            for pattern in patterns:
                if fnmatch.fnmatch(name,pattern):
                    # multilevelCode_filePath_dict[ os.path.basename(path) ] = os.path.join(path, name)
                    filtered_files.append(os.path.join(path, name))
                    break
        if single_level:
            break
    return filtered_files

class BasicDatas(object):
	"""docstring for BasicDatas"""
	
	@classmethod
	def _write_train_corpus(self, first_type_code, second_type_code, train_contents_fe):
		outfile_path = os.path.join(CORPUS_PATH, 
									first_type_code, 
									second_type_code, 
									first_type_code+'.'+second_type_code+'.train')
		print outfile_path
		fout = open(outfile_path ,'w')
		fout.write('\n'.join( sorted(train_contents_fe)) )
		fout.close()

	@classmethod
	def preprocess(self, patterns_csv, access_database_permission = False):
		# 不从数据库中拿数据，用本地的csv文件（相当于从数据库中拿）
		if access_database_permission == False:
			filtered_files = traverse_directory_tree(CORPUS_PATH, patterns=patterns_csv)
			filtered_files = sorted(filtered_files)
			for csvfile in filtered_files:
				print csvfile
				# target_dirname = os.path.dirname(csvfile)
				first_type_code,second_type_code,_ = os.path.basename( csvfile ).split('.')
				self._write_train_corpus(first_type_code, 
										second_type_code, 
										PreprocessingUtil.preprocess_source_file(csvfile))
			return True

	@classmethod
	def preprocessing_basic_datas(patterns_csv,access_database_permission = False):
		# 从数据库中拿数据，生成本地的csv文件
		second_type_code_list = orm.session.query( \
			orm.distinct(orm.BasicData.second_type_code)). \
			order_by(orm.BasicData.second_type_code)

		for item in second_type_code_list:
			second_type_code = item[0]
			assert(second_type_code.isdigit())

			first_type_code = second_type_code[:2]
			if first_type_code == '00':
				continue
			lv2_code = second_type_code
			print second_type_code

			product_list = orm.session.query(orm.BasicData.name,
									orm.BasicData.unit,
									orm.BasicData.brand,
									orm.BasicData.spec,
									orm.BasicData.attrs).filter(orm.BasicData.second_type_code==lv2_code).all()
			contents = list()
			for item in product_list:
				product = Product(item.name,item.unit,item.brand,item.spec,item.attrs)
				contents.append(product)
			
			current_path = corpus_path
			for level_code in [first_type_code,second_type_code]:
				current_path = os.path.join(current_path,level_code)
				if not os.path.exists(current_path):
					os.mkdir(current_path)

			# --------------------------------------------------------------
			csv_file = os.path.join(corpus_path, current_path, first_type_code+'.'+second_type_code+'.csv')
			print csv_file
			writer = csv.writer(file(csv_file, 'wb'), quoting=csv.QUOTE_ALL)
			header = ['name', 'unit', 'brand', 'spec', 'attrs']
			writer.writerow(header)
			for product in contents:
				writer.writerow(product)



def main():
	print "usage: python preprocessing_unrefined 01 or python preprocessing_unrefined"
	filtered_conditions = sys.argv[1]
	print "sys.argv: ",sys.argv
	print "filtered_conditions:",filtered_conditions

	para = (filtered_conditions+'.*.csv') if filtered_conditions else ('*.csv')
	BasicDatas.preprocess(para)#01 25

if __name__ == '__main__':
	main()
