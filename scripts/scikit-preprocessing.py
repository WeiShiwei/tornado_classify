#!/usr/bin/python
# #_*_ coding: utf-8 _*_


import os,sys
import collections

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../'))
from glearn.classify import orm
from glearn.classify import global_variables as gv
from glearn.feature_engineering import Feature_Engineering

reload(sys)                         #  
sys.setdefaultencoding('utf-8')     # 


corpus_path = gv.CORPUS_PATH
Product = collections.namedtuple("Product","name unit brand spec attrs")
escape_lv2_codes = ['0001',
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

def fetch_content(product_tuple):
	name = product_tuple.name.strip() if product_tuple.name is not None else ''
	unit = product_tuple.unit.strip()[:0] if product_tuple.unit is not None else ''
	brand =product_tuple.brand.strip()[:0] if product_tuple.brand is not None else ''
	spec = product_tuple.spec.strip() if product_tuple.spec is not None else '' #可以抽取‘型号’、正则表达去掉‘数字’等等，暂时不作处理
	attrs = product_tuple.attrs.strip() if product_tuple.attrs is not None else ''

	temp_list = [name,unit,brand,spec,attrs]
	content = ' '.join(temp_list)
	return content.strip()


def main():
	# import pdb;pdb.set_trace()
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
			content = fetch_content(product)
			contents.append(content)
		
		current_path = corpus_path
		for level_code in [first_type_code,second_type_code]:
			current_path = os.path.join(current_path,level_code)
			if not os.path.exists(current_path):
				os.mkdir(current_path)
		

		outfile_path = os.path.join(corpus_path,current_path, first_type_code+'.'+second_type_code+'.txt')
		fout = open(outfile_path ,'w')
		fout.write('\n'.join(sorted(contents)))
		fout.close()


		contents = Feature_Engineering.normalize_lines(contents) ###
		contents_set = set(contents)
		contents = list(contents_set)


		outfile_path = os.path.join(corpus_path, current_path, first_type_code+'.'+second_type_code+'.train')
		fout = open(outfile_path ,'w')
		fout.write('\n'.join(sorted(contents)))
		fout.close()

if __name__ == '__main__':
	main()
