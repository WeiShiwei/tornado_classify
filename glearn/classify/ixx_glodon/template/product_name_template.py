# coding=utf-8

import os
import sys
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '..'))

from xlrd import open_workbook
# from template.match_rule import *
# from extractor.data_extractor import DataExtractor
# from extractor.data_extractor import template_redis
from model.basic_element import BasicElement
from model.base_material_type import *
from model.base_material_type_attr import *
from model.base_material_type_attr_value import *
from model.base_material_type_attr_key_word import *
from model.base_material_type_attr_rule import *
from model.session import *
from template.logger import logger ###20150319
# from logger import logger
from api.config import config
from sqlalchemy.orm import aliased

from model.base_material_type_attr_sets import *
from model.attr_set_values import *

import math


from model.base_material_type_product_names import *


class ProductNameTemplate(object):
	"""docstring for ProductNameTemplate"""

	@staticmethod
	def retrieve_templates(category):
		product_name_list = list()

		first_type_code,second_type_code = category.split('.')
		second_type_code = first_type_code+second_type_code

		logger.debug('retrieve (%s,%s)\'s lv2 product names template'%(first_type_code,second_type_code))
		with get_session() as session:
			records = session.query(BaseMaterialTypeProductNames.description).filter(
				BaseMaterialTypeProductNames.first_type_code==first_type_code,
				BaseMaterialTypeProductNames.second_type_code==second_type_code).all()
			for r in records:
				product_name_list.append( r.description )
		return product_name_list