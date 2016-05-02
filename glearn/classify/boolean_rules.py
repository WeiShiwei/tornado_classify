#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import collections
import itertools
import glob
from time import time
from datetime import datetime
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../../dependence'))

from distutils.util import get_platform
sys.path.insert(0, "ahocorasick-0.9/build/lib.%s-%s" % (get_platform(), sys.version[0:3]))
import ahocorasick
import ahocorasick.graphviz

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../..'))
from glearn.specification.template import SpecificationTemplate,ProductNameTemplate
from glearn.classify.orm import BaseMaterialType
from glearn.classify import global_variables as gv
from glearn.rule.boolean_rule_parser import BooleanRuleParser,LOGICAL_AND_SYMBOL,LOGICAL_OR_SYMBOL,LOGICAL_NOT_SYMBOL

import orm
reload(sys)                         #  
sys.setdefaultencoding('utf-8')     # 

class BooleanRulesClassifier(object):
	"""docstring for Boolean_Rules"""

	def __init__(self):
		super(BooleanRulesClassifier, self).__init__()
		self.lv2_codes = BaseMaterialType.all_lv2_codes()

		self.booleanRule_count = 0
		self.booleanRule_updatedTime = datetime(2000, 8, 6, 6, 29, 51, 144126)

	def __load_exclusive_features(self):
		""" 排他性特征 """
		print('\n'+'=' * 80)
		print("Ahocorasick KeywordTree Making: ")
		t0 = time()
		# ----------------------------------------
		self.__load_key_word_references()
		# self.__load_specification_templates()
		# self.__load_product_templates() ###产品名称有错误的数据，接口已写完20150401
		# ----------------------------------------

		print("making time: %0.3fs" % (time() - t0))

	def __load_key_word_references(self):
		self.tree_keyWord = ahocorasick.KeywordTree()
		all_key_word_refereces = orm.session.query( orm.KeyWordReference.first_type_code, orm.KeyWordReference.second_type_code,
											orm.KeyWordReference.name).\
											all()
		self.all_key_words = list()
		self.keyWord_secCode_dict = collections.defaultdict(str)
		for kwr in all_key_word_refereces:
			lv2_label = kwr.second_type_code
			if lv2_label == '':
				continue
			name = kwr.name.strip().upper() 
			# -----------------------------------
			rule_desc = name.replace('&',LOGICAL_AND_SYMBOL)
			intermediate_rules = BooleanRuleParser.parse(rule_desc)
			for irule in intermediate_rules:
				items = irule.split( LOGICAL_AND_SYMBOL )
				if len(items)==1:
					# 母线槽
					self.all_key_words.append(irule)
					self.keyWord_secCode_dict[irule] = lv2_label
				else:
					# MPa∧标准块
					for item in items:
						self.all_key_words.append(item)
					self.all_key_words.append( irule )
					self.keyWord_secCode_dict[ irule ] = lv2_label
			# ------------------------------------------
		pn_files = glob.glob( os.path.join(gv.RESOURCES_PRODUCT_PATH, '*.pn') )
		for pn_file in pn_files:
			lv2_label = os.path.basename(pn_file).strip('.pn')
			with open(pn_file) as infile:
				names = [line.strip() for line in infile.readlines()]
				for name in names:
					name = unicode(name) ###从数据里面拿来的文本数据是unicode的,从文本文件拿来的文本数据是utf-8
					self.all_key_words.append(name)
					self.keyWord_secCode_dict[name] = lv2_label
		# ------------------------------------------

		for word in self.all_key_words:
			self.tree_keyWord.add(word)
		self.tree_keyWord.make()

	def __load_specification_templates(self):
		self.specTemplate_secCode_dict = collections.defaultdict(str)

		self.all_spec_templates = list()
		for lv2_code in sorted(self.lv2_codes):
			category = '.'.join( [lv2_code[0:2], lv2_code[2:]] )
			try:
				templates,from_redis = SpecificationTemplate.retrieve_templates(category)
			except Exception, e:
				# TypeError: 'NoneType' object is not iterable
				continue
				# raise e
			
			# print templates
			self.all_spec_templates.extend(templates)
			
			lv2_code = ''.join( category.split('.') )
			for template in templates:
				self.specTemplate_secCode_dict[template] = lv2_code

		self.tree_specTemplate.add('!@#$%^&*')
		for template in self.all_spec_templates:
			self.tree_specTemplate.add(template)
		self.tree_specTemplate.make()

	def __load_product_templates(self):
		self.productTemplate_secCode_dict = collections.defaultdict(str)

		self.all_product_templates = list()
		for lv2_code in sorted(self.lv2_codes):
			category = '.'.join( [lv2_code[0:2], lv2_code[2:]] )
			try:
				templates,from_redis = ProductNameTemplate.retrieve_templates(category)
				print templates
			except Exception, e:
				print e
				continue
			self.all_product_templates.extend(templates)
			
			lv2_code = ''.join( category.split('.') )
			for template in templates:
				self.productTemplate_secCode_dict[template] = lv2_code		

		self.tree_productTemplate.add('!@#$%^&*')
		for template in self.all_product_templates:
			self.tree_productTemplate.add(template)
		self.tree_productTemplate.make()


	def multi_pattern_parse(self, multi_patterns):
		""" 多模式串的解析 
		example:
		sentence: NY150耐油石棉橡胶板   厚×宽：1~80mm×500~3000mm 厚度δ（）:1~80;品种:石棉橡胶板
		multi patterns: 油石,橡胶板,石棉,橡胶板
		votes: defaultdict(<type 'str'>, {u'0325': 1, u'1301': 1, u'0201': 2})
		
		排列组合
		import itertools  
		>>> print list(itertools.permutations([1,2,3,4],2))  
		[(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 4), (4, 1), (4, 2), (4, 3)] 
		>>> print list(itertools.combinations([1,2,3,4],2))  
		[(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)] 
		"""
		answer = ''
		sorted_multi_patterns = sorted( set(multi_patterns) )
		sorted_multi_patterns_combs = list()
		sorted_multi_patterns_combs.extend(itertools.combinations( sorted_multi_patterns,3))
		sorted_multi_patterns_combs.extend(itertools.combinations( sorted_multi_patterns,2))
		for comb in sorted_multi_patterns_combs:
			try:
				answer = self.keyWord_secCode_dict[ LOGICAL_AND_SYMBOL.join(comb) ] # 橡胶板&油石&石棉
				# print "multi pattern parse:",answer # DEBUG
				if answer: 
					return answer
			except Exception, e:
				answer = ''
		return answer

	def multi_pattern_matching(self, doc):
		""" 多模式串的匹配 """
		multi_patterns = list()

		votes_dict = collections.defaultdict(str)
		ordered_code_list = list()
		# --------词库的匹配----------------
		for match in self.tree_keyWord.findall_long(doc): # findall_long
			word_matched = unicode(doc[int(match[0]):int(match[1])].strip())
			multi_patterns.append(word_matched)
			code = self.keyWord_secCode_dict[word_matched]
			if code == '':
				continue
			if code not in ordered_code_list:
				ordered_code_list.append(code)
			votes_dict[code] = votes_dict.get(code, 0) + 2

		# -------型号数据的匹配-------------
		"""
		for match in self.tree_specTemplate.findall_long(doc): # findall_long
			word_matched = unicode(doc[int(match[0]):int(match[1])].strip())
			multi_patterns.append(word_matched)
			code = self.specTemplate_secCode_dict[word_matched]
			if code not in ordered_code_list:
				ordered_code_list.append(code)
			# 型号字符串有很短的字符串，可能会与其他类别的数据刚好匹配上，所以其的投票权重仅为词库数据的50%
			votes_dict[code] = votes_dict.get(code, 0) + 1 
		"""

		# -------产品名称数据的匹配----------
		# for match in self.tree_productTemplate.findall_long(doc): # findall_long
		# 	word_matched = unicode(doc[int(match[0]):int(match[1])].strip())
		# 	multi_patterns.append(word_matched)
		# 	code = self.productTemplate_secCode_dict[word_matched]
		# 	if code not in ordered_code_list:
		# 		ordered_code_list.append(code)
		# 	votes_dict[code] = votes_dict.get(code, 0) + 1
		# print '产品名称数据的匹配:',' '.join(multi_patterns)
		# -------产品名称数据的匹配----------
		
		print '#multi patterns#:',','.join(multi_patterns) # DEBUG
		# print '#votes#:',votes_dict # DEBUG
		# -------------------------------------------------------------------
		code_most_likely = ''
		# 多模式匹配分析(排序/组合/查询)
		answer_refer = self.multi_pattern_parse(multi_patterns)
		if answer_refer:
			code_most_likely = answer_refer
			return code_most_likely,' '.join(multi_patterns)

		# 如果投票列表为空
		if not ordered_code_list:
			return code_most_likely,' '.join(multi_patterns)

		# 如果只有一种类型的投票
		if len(ordered_code_list) == 1:
			code_most_likely = ordered_code_list[0]
			return code_most_likely,' '.join(multi_patterns)

		# 多模式匹配分析无结果
		ordered_code_count_list = list()
		for code in ordered_code_list:
			votes = votes_dict[code]
			ordered_code_count_list.append(votes)
		# print "ordered_code_count_list: ",ordered_code_count_list
		max_votes = max(ordered_code_count_list)
		if ordered_code_count_list.count(max_votes)>1:
			# 如果最大投票数的code对应多个，则code_most_likely被赋值为''，交给分类器去预测
			return '',' '.join(multi_patterns)
		else:
			# 投票数最多的类别即为最终预测类别
			idx = ordered_code_count_list.index(max_votes)
			code_most_likely = ordered_code_list[idx]
			return code_most_likely,' '.join(multi_patterns)

	def predict(self, docs):
		keyWordReference_updatedTime,keyWordReference_Count = orm.KeyWordReference.fetch_latest_updated_time()
		if keyWordReference_updatedTime != self.booleanRule_updatedTime or self.booleanRule_count!=int(keyWordReference_Count):
			print "keyWordReference_updatedTime: ",keyWordReference_updatedTime
			print "booleanRule_updatedTime: ",self.booleanRule_updatedTime
			print "keyWordReference_Count: ",keyWordReference_Count
			print "booleanRule_count: ",self.booleanRule_count
			self.__load_exclusive_features()
			self.booleanRule_updatedTime = keyWordReference_updatedTime
			self.booleanRule_count = keyWordReference_Count

		boolean_rules_res = list();multi_patterns_res=list()
		for doc in docs:
			doc = doc.encode('utf-8')
			doc = doc.upper()
			if doc == '':
				boolean_rules_res.append('null')
				continue
			code_most_likely,multi_patterns = self.multi_pattern_matching(doc)
			boolean_rules_res.append(code_most_likely)
			multi_patterns_res.append( multi_patterns )
		return boolean_rules_res,multi_patterns_res


def main():
	t0 = time()
	boolean_rules_clf = BooleanRulesClassifier()
	print("BooleanRulesClassifier Instantiated done in %fs" % (time() - t0))

	docs = [
	"上海勃展阀门制造有限公司（太原办事处） 焊接球阀 Q61PPL-16C 勃展 台 220 备注：DN500以上规格价格另报，",
	"NY150耐油石棉橡胶板   厚×宽：1~80mm×500~3000mm 厚度δ（）:1~80;品种:石棉橡胶板 xiuashi石棉",
	"铝合金扶手	0.45/0.75kV 54×1.5",
	"铜芯交联聚氯乙烯绝缘电力电缆ZC-YJY-73	21/35kV 1×185",
	"https://www.baidu.com/"
	]
	t0 = time()
	boolean_rules_res,multi_patterns_res = boolean_rules_clf.predict(docs)
	print boolean_rules_res
	print multi_patterns_res
	print("boolean_rules_clf predict done in %fs" % (time() - t0))

if __name__ == "__main__":
	main()
