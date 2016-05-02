#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: WeiShiwei <weishiwei920@163.com>
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
# import matplotlib.pyplot as plt

import os
# import time
import collections

from sklearn.utils.extmath import density
from sklearn import metrics


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

###############################################################################
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '..'))

from base_clf import BaseClf
from feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.externals import joblib

class LibSVMClassifier(BaseClf):
	"""docstring for LibSVMClassifier"""

	def __init__(self, clf=svm.SVC(kernel='linear',probability=True), 
		vtr=TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english',ngram_range=(1, 2),token_pattern=r'\b\w+\b')):
		super(LibSVMClassifier, self).__init__(clf, vtr)
	
	def predict_categorys(self,pred):
		return [self._categories[cate_index] for cate_index in pred]

	def fetch_category(self,pred):
		""" pred是self._categories的索引"""
		return self._categories[pred]

	def dump(self,file_name):
		# file_name = os.path.join(gv.models_path,multilevel_code)+'/'+'_'.join(data_train.target_names)+'.joblib'
		return joblib.dump(self, file_name)

	@classmethod
	def load(self,file_name):
		return joblib.load(file_name)

	def score(self, y_test, pred):
	    """score的应用场景应该是有新的测试样本，该样本应该独立于模型的训练样本
	    模型的评价大致有两种方式：一、独立的测试样本；二、交叉验证;
	    """
	    print_top10 = True
	    print_report = True
	    print_cm = True
	    
	    categories = self._categories
	    # if type(vectorizer) == type(feature_extraction.TfidfVectorizer.TfidfVectorizer):
	    feature_names = np.asarray(self._vtr.get_feature_names())
	    # else:
	    #     feature_names = None 
	    # import pdb;pdb.set_trace()
	    score = metrics.accuracy_score(y_test, pred) # 模型的评价accuracy_score/recall_score
	    print("accuracy_score:   %0.3f" % score)
	    score = metrics.recall_score(y_test, pred) # 模型的评价accuracy_score/recall_score
	    print("recall_score:   %0.3f" % score)
	    score = metrics.f1_score(y_test, pred) # 模型的评价accuracy_score/recall_score
	    print("f1-score:   %0.3f" % score)

	    import pdb;pdb.set_trace()
	    if hasattr(self._clf, 'coef_'):
	        print("dimensionality: %d" % self._clf.coef_.shape[1])
	        print("density: %f" % density(self._clf.coef_))
	        if print_top10 and feature_names is not None:
	            print("top 10 keywords per class:")
	            for i, category in enumerate(categories):
	                top10 = np.argsort(self._clf.coef_[i])[-100:]
	                # print(trim("%s: %s"% (category, " ".join(feature_names[top10]))))
	                print("%s: %s"% (category, " ".join(feature_names[top10])))
	        print()

	    # import pdb;pdb.set_trace()
	    if print_report:
	        print("classification report:")
	        print(metrics.classification_report(y_test, pred,target_names=categories))
	    # import pdb;pdb.set_trace()
	    if print_cm:
	        print("confusion matrix:")
	        print(metrics.confusion_matrix(y_test, pred))

	    print()
	    
	# def predict_proba(self, docs):
	# 	""" docs既可以是列表格式的文档集，又可以是Bunch格式的文档集
	# 	"""
	# 	vectorizer_ = self._get_vectorizer_()
	# 	if type(docs) == type(list()):
	# 		# 列表格式的文档集
	# 		X_test = vectorizer_.fit_transform(docs)
	# 	else:
	# 		# Bunch格式的文档集
	# 		X_test = vectorizer_.fit_transform(docs.data)
	# 	X_test = self._chi2.transform(X_test) #

	# 	print('_' * 80)
	# 	print("Predicting: ")
	# 	print(self._clf)
	# 	t0 = time()
	# 	import pdb;pdb.set_trace()
	# 	proba = self._clf.predict_proba(X_test)
	# 	test_time = time() - t0
	# 	print("test time:  %0.3fs" % test_time)
	# 	return proba

from sklearn import cross_validation
from datasets import load_files
from sklearn.externals import joblib
import os
import global_variables as gv

from sklearn.utils import shuffle
from linear_svc import LinearSVCClassifier

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '..'))
from feature_extraction.text import TfidfVectorizer
from feature_selection import chi2

from _weights import _balance_weights

def main():
	libsvm_svc_clf = LibSVMClassifier()
	# --------------------------------------------------
	multilevel_code = '04'#25
	# # categories_select = ['2501','2505','2521','2523','2525','2527','2529','2531','2541','2543']
	# # categories_select = ['2501','2505','2521','2523','2525','2527']
	# # categories_select = ['0103','0105','0107','0109','0111']
	categories_select = None
	data_train = load_files.fetch_Bunch_datas(multilevel_code,categories=categories_select)

	# -------------------fit_transform-------------------------------
	X_train = libsvm_svc_clf.transform(data_train)
	y_train = data_train.target
	#\ X_train = chi2.fit_transform(X_train, y_train)

	X, y = shuffle(X_train, y_train, random_state=13)
	offset = int(X.shape[0] * 0.8)
	X_train, y_train = X[:offset], y[:offset]
	X_test, y_test = X[offset:], y[offset:]

	t0 = time()
	sample_weight = _balance_weights(y_train)
	libsvm_svc_clf.fit(X_train, y_train)
	print("done in %fs" % (time() - t0))


	pred = libsvm_svc_clf._clf.predict(X_test)
	libsvm_svc_clf.score( y_test, pred)
	import pdb;pdb.set_trace()

# -------------------------------------------------------------------------------------------------
	# \ 不同大类的概率分布
	# multilevel_code = '05'#25
	# categories_select = None
	# data_train_01 = load_files.fetch_Bunch_datas(multilevel_code,categories=categories_select)
	# # X_train_01 = libsvm_svc_clf.transform(data_train_01)
	# y_train_01 = data_train_01.target
	# # print (clf.score(X_train_01, y_train_01))

	# docs = data_train_01.data
	# proba = libsvm_svc_clf.predict_proba(docs)
	# proba_list = proba.tolist()
	# # m = max(proba_list[0])
	# count = 0
	# for l in proba_list:
	# 	if max(l)>0.9:
	# 		print (l)
	# 		count += 1
	# 		# m = max(l)
	# print (float(count)/len(proba_list))
	# import pdb;pdb.set_trace()



	# clf = libsvm_svc_clf._clf
	# scores = cross_validation.cross_val_score( clf, X_train, data_train.target, cv=5)
	# scores
	# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# -------------------------------------------------------------------------------------------------
	# file_name = os.path.join(gv.models_path,multilevel_code)+'/'+'_'.join(data_train.target_names)+'.joblib'
	# libsvm_svc_clf.dump(file_name)
# # -------------------------------------------------------------------------------------------------
	import pdb;pdb.set_trace()
	linear_svc_clf_ = LinearSVCClassifier.load("/home/weishiwei/800w_classifier/tornado_classify/glearn/models/00_01_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_50_51_52_53_54_55_56_57_58_80_98_99.joblib")
	libsvm_svc_clf_ = libsvm_svc_clf.load("/home/weishiwei/800w_classifier/tornado_classify/glearn/models/30/3001_3005_3007_3009_3011_3013_3021_3023_3025_3031_3033_3035_3039_3041.joblib")
	
	multilevel_code = '30'
	# categories_select = ['3001']
	categories_select = None
	data_test = load_files.fetch_Bunch_datas(multilevel_code,categories=categories_select)

	# import pdb;pdb.set_trace()
	# pred = linear_svc_clf_.predict(data_test)
	# predict_res = linear_svc_clf_.predict_categorys(pred)
	# # y_test = linear_svc_clf_.y_test_reindexing(data_test)
	# # linear_svc_clf_.score( y_test, pred)
	# i = 0 
	# for res in predict_res:
	# 	if res == '30':
	# 		i = i+1
	# print (i)
	# print (len(predict_res))

	import pdb;pdb.set_trace()
	pred = libsvm_svc_clf_.predict(data_test)
	proba = libsvm_svc_clf_.predict_proba(data_test)
	libsvm_svc_clf_.score(data_test.target,pred)










if __name__ == "__main__":
	main()
	# pass