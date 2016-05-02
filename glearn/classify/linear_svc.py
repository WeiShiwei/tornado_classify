#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: WeiShiwei <weishiwei920@163.com>
# License: BSD 3 clause

from __future__ import print_function

import numpy as np
import sys
from time import time

import os
import collections
from sklearn import metrics
from sklearn.utils.extmath import density
from sklearn.externals import joblib
###############################################################################
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../..'))

from glearn.classify.base_clf import BaseClf
from glearn.classify.sklearn_classifier import LinearSVC
from glearn.feature_extraction.text import TfidfVectorizer

from _stop_words import stop_words

class LinearSVCClassifier(BaseClf):
	"""docstring for LinearSVCClassifier"""

	# class_weight='auto'可以一定程度上控制样本不平衡的问题
	def __init__(self, clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3,class_weight='auto'), 
		vtr = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=stop_words,encoding='utf-8',
							ngram_range=(1, 2),token_pattern='(?u)\\b\\w\\w+\\b',
							max_features=80000)):
	# def __init__(self, clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3,class_weight='auto'), 
	# 	vtr = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=stop_words,encoding='utf-8',
	# 						max_features=80000)):
		super(LinearSVCClassifier, self).__init__(clf, vtr)
	
	def fetch_categorys(self,pred):
		return [self._categories[cate_index] for cate_index in pred]

	def fetch_category(self,pred):
		""" pred是self._categories的索引"""
		return self._categories[pred]

	def dump(self,file_name):
		return joblib.dump(self, file_name)

	@classmethod
	def load(self,file_name):
		return joblib.load(file_name)

	def __fetch_prior_probability(self, hyperplane_distance):
		return hyperplane_distance

	def predict_proba(self, docs):
		""" docs既可以是列表格式的文档集，又可以是Bunch格式的文档集
		重写父类方法
		"""
		hyperplane_distance = self.decision_function(docs)
		return self.__fetch_prior_probability(hyperplane_distance)

	def decision_function(self, docs):
		""" docs既可以是列表格式的文档集，又可以是Bunch格式的文档集
		重写父类方法
		"""
		vectorizer_ = self.vectorizer_
		if isinstance(docs, list):# 列表格式的文档集
			#\ X_test = vectorizer_.fit_transform(docs) # fit_transform是依据当前的docs重新矢量化转换
			X_test = vectorizer_.transform(docs) # fit_transform是依据vectorizer_的vocabulary_进行矢量化转换
		elif isinstance(docs, Bunch):# Bunch格式的文档集
			#\ <class 'datasets.load_files.Bunch'>
			#\ X_test = vectorizer_.fit_transform(docs.data)
			X_test = vectorizer_.transform(docs.data)
		elif isinstance(docs, scipy.sparse.csr.csr_matrix):
			X_test = docs
		else:
			X_test = None

		print('_' * 80)
		print("Predicting: ")
		print(self._clf)
		t0 = time()
		hyperplane_distance = self._clf.decision_function(X_test)
		test_time = time() - t0
		print("test time:  %0.3fs" % test_time)
		return hyperplane_distance
		# return self.__fetch_prior_probability(hyperplane_distance)


def _plot_learning_curve(multilevel_code=''):
	t0 = time()
	linear_svc_clf = LinearSVCClassifier()
	# -------------------------------------------------------------------------------------------------
	categories_select = None
	if multilevel_code:
		data_train = load_files( os.path.join(gv.CORPUS_PATH, multilevel_code),
	                            encoding='utf-8',
	                            load_files_type='line')
	else:
		# 训练第一层模型，用于预测一级类
		data_train = load_files_bunch.fetch_Bunch_datas(multilevel_code,categories=categories_select)
	# import pdb;pdb.set_trace()
	# -------------------------------------------------------------------------------------------------
	X_train = linear_svc_clf.transform(data_train)
	y_train = data_train.target

	# plot_learning_curve #
	X, y = shuffle(X_train, y_train, random_state=13)
	title = "Learning Curves (LinearSVC)"
	# Cross validation with 100 iterations to get smoother mean test and train
	# score curves, each time with 20% data randomly selected as a validation set.
	cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10,
										test_size=0.2, random_state=0)
	estimator = linear_svc_clf._clf
	plt = plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
	# plt.show()
	
	return plt


def main():
	t0 = time()
	linear_svc_clf = LinearSVCClassifier()
	# -------------------------------------------------------------------------------------------------
	multilevel_code = '' #01
	categories_select = None
	# data_train = load_files( os.path.join(gv.CORPUS_PATH, multilevel_code),
 #                            encoding='utf-8',
 #                            load_files_type='line')
	data_train = load_files.fetch_Bunch_datas(multilevel_code,categories=categories_select)
	# import pdb;pdb.set_trace()
	# -------------------------------------------------------------------------------------------------
	X_train = linear_svc_clf.transform(data_train)
	y_train = data_train.target

	X, y = shuffle(X_train, y_train, random_state=13)
	offset = int(X.shape[0] * 0.8)
	X_train, y_train = X[:offset], y[:offset]
	X_test, y_test = X[offset:], y[offset:]

	linear_svc_clf.fit(X_train, y_train)

	pred = linear_svc_clf._clf.predict(X_test)
	linear_svc_clf.score(y_test,pred)
	#\ -------------------------------------------------------------------------------------------------
	#\ print("done in %fs" % (time() - t0))
	#\ # import pdb;pdb.set_trace()
	#\ file_name = os.path.join(gv.models_path,multilevel_code)+'/'+'_'.join(data_train.target_names)+'.joblib'
	#\ linear_svc_clf.dump(file_name)
	#\ print("done in %fs" % (time() - t0))
	#\ linear_svc_clf = joblib.load('/home/weishiwei/800w_classifier/tornado_classify/glearn/models/00_01_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_50_51_52_53_54_55_56_57_58_80_98_99.joblib')

	# cross_validation #
	clf = linear_svc_clf._clf
	scores = cross_validation.cross_val_score( clf, X_train, y_train, cv=5)
	print(scores)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == "__main__":
	from sklearn.externals import joblib
	from sklearn import cross_validation
	from sklearn.utils import shuffle
	#\ from glearn.feature_selection import chi2
	from glearn.datasets import load_files as load_files_bunch
	from glearn.datasets.load_files_platform import load_files # 训练一级类模型是有问题的
	from glearn.classify import global_variables as gv
	from glearn.model_selection.plot_learning_curve import plot_learning_curve
	
	# -------------------------------------------------------
	# main()zz


	# ------------------------------------------------------- #
	print("usage: python linear_svc.py 01 | python linear_svc.py")
	print(sys.argv)
	if len(sys.argv)>1:
		multilevel_code = sys.argv[1]
	else:
		multilevel_code = ''

	gv.CORPUS_PATH = os.path.expanduser( os.path.join( '~','scikit_learn_data_contrast' ) )
	plt = _plot_learning_curve(multilevel_code=multilevel_code)
	plt.savefig( 
		os.path.join(
			os.path.expanduser( os.path.join( '~','Documents' ) ),
			'figure_'+multilevel_code+'_orgin.png'),
		dpi = 200)

	gv.CORPUS_PATH = os.path.expanduser( os.path.join( '~','scikit_learn_data' ) )
	plt = _plot_learning_curve(multilevel_code=multilevel_code)

	plt.savefig( 
		os.path.join(
			os.path.expanduser( os.path.join( '~','Documents' ) ),
			'figure_'+multilevel_code+'_refine.png'),
		dpi = 200)
