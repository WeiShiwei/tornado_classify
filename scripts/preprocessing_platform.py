#!/usr/bin/python
# #_*_ coding: utf-8 _*_


import os,sys
import collections
import commands
import shutil

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../'))
from glearn.classify import orm
from glearn.classify import global_variables as gv
from glearn.feature_engineering import Feature_Engineering


Product = collections.namedtuple("Product","name unit brand spec attrs")
def fetch_content(product_tuple):
	name = product_tuple.name.strip() if product_tuple.name is not None else ''
	unit = product_tuple.unit.strip()[:0] if product_tuple.unit is not None else ''
	brand =product_tuple.brand.strip()[:0] if product_tuple.brand is not None else ''
	spec = product_tuple.spec.strip() if product_tuple.spec is not None else '' #可以抽取‘型号’、正则表达去掉‘数字’等等，暂时不作处理
	attrs = product_tuple.attrs.strip() if product_tuple.attrs is not None else ''

	temp_list = [name,unit,brand,spec,attrs]
	content = ' '.join(temp_list)
	return content.strip()

ARCHIVE_NAME = gv.gldjc_archive
GLDJC_PATH = os.path.join(gv.get_data_home(), 'gldjc')
GLDJC_ARCHIVE_PATH = os.path.join(GLDJC_PATH, ARCHIVE_NAME )
GLDJC_ARCHIVE_FILE_PATH = os.path.join(GLDJC_PATH, ARCHIVE_NAME+'.tar.gz' )

def main():
	reload(sys)                         #  
	sys.setdefaultencoding('utf-8')     # 

	# ----------------------------------------------------------------------------------
	if os.path.exists( GLDJC_ARCHIVE_PATH ):
		shutil.rmtree( GLDJC_ARCHIVE_PATH )

	# ----------------------------------------------------------------------------------
	# import pdb;pdb.set_trace()
	second_type_code_list = orm.session.query( \
		orm.distinct(orm.BasicData.second_type_code)). \
		order_by(orm.BasicData.second_type_code)

	for item in second_type_code_list:
		second_type_code = item[0]
		print second_type_code
		# assert(second_type_code) ==  '****'
		assert(second_type_code.isdigit())
		first_type_code = second_type_code[:2]
		# second_type_code = second_type_code[-2:]
		
		lv2_code = second_type_code
		product_list = orm.session.query(orm.BasicData.name,
								orm.BasicData.unit,
								orm.BasicData.brand,
								orm.BasicData.spec,
								orm.BasicData.attrs).filter(orm.BasicData.second_type_code==lv2_code).all()
		contents = list()
		for item in product_list:
			product = Product(item.name,item.unit,item.brand,item.spec,item.attrs)
			content = fetch_content(product)
			contents.append(content)


		contents = Feature_Engineering.normalize_lines(contents) ###
		contents_set = set(contents)
		contents = list(contents_set)

		gldjc_path = os.path.join(GLDJC_ARCHIVE_PATH ,'ml-gldjc-com-train' ,'.'.join([first_type_code,second_type_code]) )
		if not os.path.exists(gldjc_path):
			os.makedirs(gldjc_path)

		outfile_path = os.path.join( gldjc_path, '.'.join([first_type_code,second_type_code,'train']) )
		fout = open(outfile_path ,'w')
		fout.write('\n'.join(sorted(contents)))
		fout.close()

	# ----------------------------------------------------------------------------------
	# commandLine = 'tar -zcvf '
	# commandLine += GLDJC_ARCHIVE_PATH+'.tar.gz '
	# commandLine += gv.gldjc_archive
	# os.chdir(GLDJC_PATH)
	# (status, output)=commands.getstatusoutput(commandLine)
	# print status, output

	# # import pdb;pdb.set_trace()
	# if status == 0:
	# 	if os.path.exists(GLDJC_ARCHIVE_PATH):
	# 		shutil.rmtree(GLDJC_ARCHIVE_PATH)



if __name__ == '__main__':
	main()
