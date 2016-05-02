#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../../'))
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../classify'))
from time import time
import collections
import copy
import pprint
import xlrd

import multiprocessing

from sklearn.externals import joblib
from glearn.datasets import load_files
from glearn.classify import global_variables as gv
from glearn.classify.naive_bayes import NaiveBayesClassifier
from glearn.classify.linear_svc import LinearSVCClassifier
from glearn.classify.libsvm_svc import LibSVMClassifier
from glearn.classify.sgd_clf import SGDClassifier
from glearn.classify.boolean_rules import BooleanRulesClassifier
from glearn.feature_engineering import Segment
from glearn.feature_engineering import Feature_Engineering
# generalization for gldjc
from glearn.classify.classify_predict_platform import AsyncJobWorker

import orm
import jieba
from datetime import datetime

class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

class ClassifyEngine(object):
	""" 模型加载的过程可以学习"操作系统虚拟内存中的典型页替换算法(OPT,LRU,FIFO,Clock) """
	# naiveBayes_updatedTime = datetime(2000, 8, 6, 6, 29, 51, 144126)
	# naive_bayes_clf = NaiveBayesClassifier() # 朴素贝叶斯分类器(常驻内存)
	boolean_rules_clf = BooleanRulesClassifier() # 布尔规则分类器(常驻内存)
	
	# base_clf = init_base_clf() # 基类分类器,运行时加载
	secondary_clfs_pool = dict() # 二级分类器池
	
	Classifiers_Limited = ['LinearSVCClassifier'] # 'LinearSVCClassifier','SGDClassifier','RandomForestClassifier'

	@classmethod
	def __load_base_clf(self):
		# 基类分类器
		print '='*50
		print 'Base LinearSVCClassifier loading'
		t0 = time()
		base_clf = joblib.load(gv.base_clf_path)
		print("done in %fs" % (time() - t0))
		return base_clf

	@classmethod
	def refresh_clfs_pool(self):
		ClassifyEngine.base_clf = init_base_clf()
		ClassifyEngine.secondary_clfs_pool = dict()

	@classmethod
	def update_secondary_clfs_pool(self, base_classify_res):
		"""according to ClassifyEngine.base_classify_res"""
		lv1_codes_existing = ClassifyEngine.secondary_clfs_pool.keys()

		joblib_pool = collections.defaultdict(dict)# 
		lv1_codes = set(base_classify_res)
		for code in lv1_codes:
			if code in lv1_codes_existing:
				continue
			joblibDir_path = os.path.join(gv.MODELS_PATH,code)
			joblib_pool[code] = load_files.traverse_directory_tree(joblibDir_path,patterns='*.joblib')

		print '-'*80
		print 'Secondary Classifiers loading'
		t0 = time()
		# clf_pool是一级code为索引的二级分类器的列表
		clf_pool = collections.defaultdict(list)
		for (code,joblib_name_path_dict) in joblib_pool.items():
		    if not joblib_name_path_dict:
		        continue
		    for (joblib_name,joblib_path) in joblib_name_path_dict.items():
		        # 加载序列化存储的模型需要花费的时间需要评估,load是一个SGDClassifier类方法,其实借用一下类的方法
		        # print joblib_name
		        try:
		        	at_pos = joblib_name.index('@')
		        except Exception, e:
		        	continue #SVC
		        classifier_name = joblib_name[:at_pos]
		        if classifier_name not in ClassifyEngine.Classifiers_Limited:
		        	continue
		        clf = SGDClassifier.load(joblib_path)
		        # 模型路径下还有SVC的模型没有名称，适时可以删除之
		        # if clf.__class__.__name__ not in ClassifyEngine.Classifiers_Limited:
					# continue
		        clf_pool[code].append( clf )
		# print 'Classifiers pool:';print clf_pool
		print("done in %fs" % (time() - t0))

		ClassifyEngine.secondary_clfs_pool.update(clf_pool)

	@classmethod
	def __vote_assistant_AB(self,A,B):
		N = len(A)
		res = ['']*N
		for i in range(N):
			if A[i] == B[i]:
				res[i] = A[i]
			else:
				res[i] = ''
		return res
	@classmethod
	def __vote_assistant_ABC(self,A,B,C):
		N = len(A)
		res = ['']*N
		for i in range(N):
			a = A[i];b=B[i];c=C[i]
			if a==b or a==c:
				res[i] = a
			elif b==c:
				res[i] = b
			else:
				res[i] = ''
		return res
	@classmethod
	def vote(self, votes_list):
		"""暂时没有处理"""
		votes_list = votes_list[:3]
		if len(votes_list) == 1:
			return votes_list[0]
		if len(votes_list) == 2:
			A = votes_list[0]
			B = votes_list[1]
			return ClassifyEngine.__vote_assistant_AB(A,B)
		if len(votes_list) == 3:
			A = votes_list[0]
			B = votes_list[1]
			C = votes_list[2]	
			return ClassifyEngine.__vote_assistant_ABC(A,B,C)		

	@classmethod
	def base_classify(self, docs):
		try:
			base_clf = ClassifyEngine.base_clf
		except AttributeError, e:
			ClassifyEngine.base_clf = ClassifyEngine.__load_base_clf()
			base_clf = ClassifyEngine.base_clf

		base_classify_res = list()
		print '='*50
		print 'Base Classifiers predicting'
		pred = base_clf.predict(docs)
		base_classify_res = base_clf.fetch_categorys(pred)
		# hyperplane_distance = ClassifyEngine.base_clf.decision_function(docs)
		# 动态语言
		# ClassifyEngine.base_classify_res = base_classify_res
		# ClassifyEngine.base_hyperplane_distance = hyperplane_distance
		# 更新二级类分类器池
		ClassifyEngine.update_secondary_clfs_pool(base_classify_res)

		return base_classify_res

	@classmethod
	def secondary_classify(self, secondary_classify_task):
		""" base_classify_res是一级分类器预测结果的列表，存储一级code
		base_hyperplane_distance是样本距离超平面的距离
		"""
		lv1_category = secondary_classify_task[0]
		docs = secondary_classify_task[1]

		secondary_classify_res = list()
		# secondary_classify_proba = list()

		print '='*50
		print 'Secondary Classifiers predicting'
		t0 = time()
		# ------------------------------------------------------------------------------
		clf_cabin = ClassifyEngine.secondary_clfs_pool.get(lv1_category,None)

		if not clf_cabin:
			secondary_classify_res.append('')
			return ['']*len(docs)

		votes_list = list()
		# 现在的策略就是多模型预测
		for clf in clf_cabin:
			pred = clf.predict(docs)
			secondary_classify_res = clf.fetch_categorys(pred)
			# hyperplane_distance = ClassifyEngine.base_clf.decision_function(docs)
			votes_list.append(secondary_classify_res)
		candidates = ClassifyEngine.vote(votes_list)	
		# print candidates
		return candidates
		# ------------------------------------------------------------------------------

	@classmethod
	def __load_keyWordReferences(self,keyWordReference_updatedTime,naiveBayes_updatedTime,lv1_code=None):
		data = list()
		target = list()

		if lv1_code: # 返回二级类(lv1_code)naive_bayes模型训练数据
			''' not used'''
			target_names = orm.KeyWordReference.fetch_target_names_lv1()
			all_key_word_refereces = orm.session.query( orm.KeyWordReference )\
									.filter(orm.KeyWordReference.first_type_code == lv1_code)\
									.filter(orm.KeyWordReference.updated_at > naiveBayes_updatedTime)\
									.all()

			for kwr in all_key_word_refereces: ### 遍历key_word_references表
				lv1_label = kwr.first_type_code
				lv2_label = kwr.second_type_code
				# vector = build_vec_base_dic( kwr.name )
				# weight_int = int( kwr.weight if kwr.weight is not None else 0 )
				# vector = [ i * weight_int for i in vector] ### vector?
				name = kwr.name

				if lv1_label:
					# data.append(' '.join(jieba.cut(name.strip())))
					data.append(name.strip()) ###
					target.append(target_names.index(lv1_label))
		else: # 返回一级类naive_bayes模型训练数据
			target_names = orm.KeyWordReference.fetch_target_names_lv2()
			all_key_word_refereces = orm.session.query( orm.KeyWordReference )\
									.filter(orm.KeyWordReference.updated_at > naiveBayes_updatedTime)\
									.all()
		
			for kwr in all_key_word_refereces: ### 遍历key_word_references表
				lv1_label = kwr.first_type_code
				lv2_label = kwr.second_type_code
				# vector = build_vec_base_dic( kwr.name )
				# weight_int = int( kwr.weight if kwr.weight is not None else 0 )
				# vector = [ i * weight_int for i in vector] ### vector?
				name = kwr.name
			
				if lv2_label and lv2_label!='3211':
					data.append(' '.join(jieba.cut(name.strip())))#"花枝直剪"=>'花枝 直剪'
					target.append(target_names.index(lv2_label))
		
		for i in range(len(data)):
			print data[i]
		return Bunch(data=data,
                     target_names=target_names,
                     target=target)
		
	@classmethod
	def naive_bayes_classify(self,docs):
		"""naiveBayes_updatedTime
		keyWordReference_updatedTime
		"""
		# Naive Bayes
		keyWordReference_updatedTime = orm.KeyWordReference.fetch_latest_updated_time()# (Pdb) type(latest_update_time) == <type 'datetime.datetime'>
		naiveBayes_updatedTime = ClassifyEngine.naiveBayes_updatedTime
		if keyWordReference_updatedTime != naiveBayes_updatedTime: # 重新训练朴素贝叶斯 ,fit or partial_fit
			data_train = ClassifyEngine.__load_keyWordReferences(keyWordReference_updatedTime,naiveBayes_updatedTime) ###			
			X_train = ClassifyEngine.naive_bayes_clf.transform(data_train)
			y_train = data_train.target
			# 相对于predict，训练的频次就很低了；
			# fit花费的时间并不多,所以启动tornado_classify实例的时候就训练贝叶斯分类器
			ClassifyEngine.naive_bayes_clf.fit(X_train,y_train)
			# -------------------------------------------------------------------			
			# 更新贝叶斯分类器的时间
			ClassifyEngine.naiveBayes_updatedTime = keyWordReference_updatedTime
		
		pred = ClassifyEngine.naive_bayes_clf.predict(docs)
		naiveBayes_classify_res = ClassifyEngine.naive_bayes_clf.fetch_categorys(pred)
		print "naiveBayes_classify_res:",naiveBayes_classify_res
		print 
		# keyWordReference_updatedTime
		# 贝叶斯分类器接受的docs逻辑上应该分词，但是不能用加载了userdict.txt的结巴分词
		# naiveBayes_classify_res = ClassifyEngine.naive_bayes_classify(docs)

	@classmethod
	def predict(self, docs, identity='gldjc'):

		boolean_rules_res,multi_patterns_res = self.boolean_rules_clf.predict(docs) 
		print "Boolean Rules predict:",boolean_rules_res
		
		docs = Segment.seg(docs)
		classify_res = AsyncJobWorker.classify(identity, docs)
		print "AsyncJobWorker predict:",classify_res

		# 布尔规则的结果更新掉相应的二级类模型预测的结果
		for i in range(len(boolean_rules_res)):
			boolean_rules_pred = boolean_rules_res[i]
			if boolean_rules_pred:
				classify_res[i] = boolean_rules_pred
		print "Ultimate predict:",classify_res

		return classify_res

	@classmethod
	def classify(self, docs):
		""" classify """
		t0 = time()
		docs_num = len(docs)

		boolean_rules_res,multi_patterns_res = self.boolean_rules_clf.predict(docs) # 引用传递
		print "Boolean Rules predict:",boolean_rules_res
		
		# docs = Segment.seg(docs)
		docs = Feature_Engineering.normalize_lines(docs)
		base_classify_res = ClassifyEngine.base_classify(docs)
		print "Base Classifier predict:",base_classify_res
		
		# -------------------------------------------------------
		# baseCode_docs_dict & baseCode_indexes_dict
		# base_classify_res = ClassifyEngine.base_classify_res
		baseCode_docs_dict = collections.defaultdict(list)
		baseCode_indexes_dict = collections.defaultdict(list)
		for i in xrange(len(base_classify_res)):
			baseCode = base_classify_res[i]
			baseCode_docs_dict[baseCode].append(docs[i])
			baseCode_indexes_dict[baseCode].append(i)

		# 构建baseCode_preds_dict
		base_classify_keys = baseCode_docs_dict.keys()
		baseCode_preds_dict = collections.defaultdict(list)
		for baseCode in base_classify_keys:
			docs = baseCode_docs_dict[baseCode]
			secondary_classify_task = (baseCode, docs)			
			baseCode_preds_dict[baseCode] = ClassifyEngine.secondary_classify(secondary_classify_task)
		
		# 返回docs顺序的predict结果
		classify_res = ['']*docs_num
		for (baseCode, indexes) in baseCode_indexes_dict.items():
			preds = baseCode_preds_dict[baseCode]
			for i in range(len(indexes)):
				classify_res[indexes[i]] = preds[i]
		print "Machine Learning predict:",classify_res
		print("==>predict done in %fs" % (time() - t0))
		# ---------------------------------------------------------
		# 布尔规则的结果更新掉相应的二级类模型预测的结果
		for i in range(len(boolean_rules_res)):
		        boolean_rules_pred = boolean_rules_res[i]
		        if boolean_rules_pred:
		                classify_res[i] = boolean_rules_pred
		print "Ultimate predict:",classify_res
		return classify_res,multi_patterns_res,boolean_rules_res,classify_res


	@staticmethod
	def classify_test( data_test ):
		""" 对测试数据进行分类测试，计算相关指标 """
		docs = data_test.data
		ytest = [data_test.target_names[idx] for idx in data_test.target]
		
		ypred,multi_patterns_res,\
		boolean_rules_res,classify_res = ClassifyEngine.classify( docs )
		
		accuracy_score,recall_score,\
		f1_score,classification_report,\
		confusion_matrix = ClassifyEngine.base_clf.measure(ytest,ypred)
		
		assert len(ytest)==len(ypred),'len(ytest)!=len(ypred)'

		predict_documents = list()
		error_predict_documents = list()
		for i in range(len(ytest)):
			predict_document = [ytest[i],
								ypred[i],
								boolean_rules_res[i],
								classify_res[i],
								multi_patterns_res[i],
								docs[i].strip()
			]
			predict_documents.append(predict_document)

			if ytest[i] != ypred[i]:
				error_predict_document = {
					"document":docs[i],
					"multi_patterns":multi_patterns_res[i],
					"ytest":ytest[i],
					"ypred":ypred[i]
				}
				error_predict_documents.append( error_predict_document )

		json_result = {
			"accuracy_score":str(accuracy_score),
			"recall_score":str(recall_score),
			"f1_score":str(f1_score),
			"classification_report":classification_report,
			"confusion_matrix":str(confusion_matrix),

			"predict_documents":predict_documents,
			"error_predict_documents":error_predict_documents
		}

		# print "accuracy_score: ",json_result["accuracy_score"]
		# print "recall_score: ",json_result["recall_score"]
		# print "f1_score: ",json_result["f1_score"]
		# print "classification_report: \n",json_result["classification_report"]
		# print "confusion_matrix: \n",json_result["confusion_matrix"]
		# for document in json_result["error_predict_documents"]:
		# 	print "="*40
		# 	print "document: ",document["document"]
		# 	print "multi_patterns: ",document["multi_patterns"]
		# 	print "ytest: ",document["ytest"]
		# 	print "ypred: ",document["ypred"]
		return json_result

	@staticmethod
	def parse_xlsx_file( xlsx_file_path ):
		"""  """
		name_workbook = os.path.basename(xlsx_file_path)
		data_workbook = xlrd.open_workbook( xlsx_file_path )
		table = data_workbook.sheet_by_index(0) #通过索引顺序获取
		nrows = table.nrows
		ncols = table.ncols
		# print 'ncols=',ncols

		data = list()
		target_names = list()
		target_names_seq = list()
		target = list()
		description = name_workbook
		
		# 是否是杂志数据 定额编码 类别编码 公司名称 产品名称 规格型号 品牌 计量单位 价格 备注 材质 说明 规格
		header = table.row_values(0)
		try:
			code_colname = u'类别编码'
			code_colidx = header.index( code_colname )

			print 'header:',' '.join( header )
			for i in range(nrows)[1:]:
				row_values = table.row_values(i)
				row_values = map(lambda x:str(x), row_values)
				target_name = row_values[ code_colidx ] #类别编码

				del row_values[ code_colidx ]
				data.append( ' '.join( row_values )) #文档
				target_names_seq.append(target_name)
			
			target_names = sorted(set(target_names_seq))
			target_name_idx_dict = dict() #构建target_name的索引
			for idx,target_name in enumerate(target_names):
				target_name_idx_dict[target_name]=idx

			for target_name in target_names_seq:
				target.append( target_name_idx_dict[target_name] )

			data_test = Bunch(data=data,
		                     target_names=target_names,
		                     target=target,
		                     DESCR=description)

			json_result = ClassifyEngine.classify_test(data_test)
			return json_result
		except Exception, e:
			raise e
		

def main():
	xlsx_file_path = os.path.join( os.path.abspath(os.path.dirname(__file__)) , 
		'../../api/test/DIANLIDIANLAN/北京北方远东线缆有限公司_罗昕逸.xlsx')
	
	json_result = ClassifyEngine.parse_xlsx_file( xlsx_file_path )


if __name__ == '__main__':
	main()