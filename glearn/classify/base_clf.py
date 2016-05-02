#!/usr/bin/python
# coding=utf-8

from __future__ import print_function

import logging
import numpy as np
import os
import sys
from time import time

import scipy
# 这里只使用了sklearn的一些评价模型的工具
# from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.externals import joblib

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../..'))
from glearn.feature_extraction.text import TfidfVectorizer
from glearn.feature_selection import chi2
from glearn.datasets.load_files import Bunch

class BaseClf(object):
	"""docstring for BaseClf"""
	
	def __init__(self, clf, vtr):
		super(BaseClf, self).__init__()
		self._clf = clf
		self._vtr = vtr
		self._chi2 = chi2
	
	def __set_categories(self, categories):
		self._categories = categories

	def __set_vectorizer_(self, vectorizer):
		""" 设置预测用的vectorizer,下一步的改进是深拷贝self._vtr;训练过程中，第一次self._vtr.fit_transform(),就可以设置调用此函数了
		latest news: self.vectorizer_ = self._vtr
		"""
		# vectorizer_ = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english',vocabulary=vectorizer.vocabulary_)
		# self.vectorizer_ = vectorizer_
		self.vectorizer_ = vectorizer
	def __get_vectorizer_(self):
		# return self.vectorizer_
		return self._vtr
	
	def __chi_squared(self, X_train, y_train):
		""" __chi_squared"""
		# import pdb;pdb.set_trace()
		if self._chi2.k is 'all':
			print ("Extracting all features by a chi-squared test")
		else:
			print("Extracting %d best features by a chi-squared test" % self._chi2.k)
		t0 = time()
		X_train = self._chi2.fit_transform(X_train, y_train) #
		print("done in %fs" % (time() - t0))
		print()
		return X_train

	def __extracting_features(self,data_train_data):
		""" __extracting_features"""
		print("Extracting features from the training dataset using a sparse vectorizer")
		t0 = time()
		X_train = self._vtr.fit_transform(data_train_data)
		self.__set_vectorizer_(self._vtr) #
		duration = time() - t0
		def size_mb(docs):
			    return sum(len(s.encode('utf-8')) for s in docs) / 1e6
		print("done in %fs at %0.3fMB/s" % (duration, size_mb(data_train_data) / duration))
		print("n_samples: %d, n_features: %d" % X_train.shape)
		# print()
		return X_train
	
	def transform(self,data_train):
		categories = data_train.target_names
		self.__set_categories(categories) 
		
		X_train = self.__extracting_features(data_train.data)
		# import pdb;pdb.set_trace()###
		# X_train = self.__chi_squared(X_train, data_train.target)###20140909:1921
		return X_train

	def fit(self, X_train, y_train, sample_weight = None):
		""" 训练接口之二（函数功能精简且灵活） ,用的应该不多
		与fit_transform相比，少了__set_categories步骤,
		"""
		print('_' * 80)
		print("Training: ")
		print(self._clf)
		t0 = time()
		
		if sample_weight is not None:
			self._clf.fit(X_train,y_train,sample_weight)
		else:
			self._clf.fit(X_train,y_train)
		train_time = time() - t0
		print("train time: %0.3fs" % train_time)

		return True

	def fit_transform(self, data_train):
		""" 训练接口之一(包括一系列的预处理过程)"""
		y_train = data_train.target
		categories = data_train.target_names
		self.__set_categories(categories) 

		# import pdb;pdb.set_trace()
		X_train = self.__extracting_features(data_train.data)
		# X_train = self.__chi_squared(X_train, y_train)

		print('_' * 80)
		print("Training: ")
		print(self._clf)
		t0 = time()
		self._clf.fit(X_train,y_train)
		train_time = time() - t0
		print("train time: %0.3fs" % train_time)
		return True


	def predict(self, docs):
		""" docs既可以是列表格式的文档集，又可以是Bunch格式的文档集
		"""
		vectorizer_ = self.__get_vectorizer_()
		if isinstance(docs, list):# 列表格式的文档集
			#\ X_test = vectorizer_.fit_transform(docs) # fit_transform是依据当前的docs重新矢量化转换
			X_test = vectorizer_.transform(docs) # fit_transform是依据vectorizer_的vocabulary_进行矢量化转换
		elif isinstance(docs, Bunch):# Bunch格式的文档集
			#\ <class 'datasets.load_files.Bunch'>
			#\ X_test = vectorizer_.fit_transform(docs.data)
			X_test = vectorizer_.transform(docs.data)
		elif isinstance(docs, scipy.sparse.csr.csr_matrix) or isinstance(docs, np.ndarray):
			X_test = docs
		else:
			X_test = None
		
		#\ X_test = self._chi2.transform(X_test) #

		print('_' * 80)
		print("Predicting: ")
		print(self._clf)
		t0 = time()
		pred = self._clf.predict(X_test)
		test_time = time() - t0
		print("test time:  %0.3fs" % test_time)
		return pred
	
	def predict_proba(self, docs):
		""" docs既可以是列表格式的文档集，又可以是Bunch格式的文档集
		"""
		vectorizer_ = self.__get_vectorizer_()
		if isinstance(docs, list):# 列表格式的文档集
			#\ X_test = vectorizer_.fit_transform(docs) # fit_transform是依据当前的docs重新矢量化转换
			X_test = vectorizer_.transform(docs) # fit_transform是依据vectorizer_的vocabulary_进行矢量化转换
		elif isinstance(docs, Bunch):# Bunch格式的文档集
			#\ <class 'datasets.load_files.Bunch'>
			#\ X_test = vectorizer_.fit_transform(docs.data)
			X_test = vectorizer_.transform(docs.data)
		elif isinstance(docs, scipy.sparse.csr.csr_matrix) or isinstance(docs, np.ndarray):
			X_test = docs
		else:
			X_test = None

		# X_test = self._chi2.transform(X_test) #

		print('_' * 80)
		print("Predicting: ")
		print(self._clf)
		t0 = time()
		proba = self._clf.predict_proba(X_test)
		test_time = time() - t0
		print("test time:  %0.3fs" % test_time)
		return proba


	def __y_test_transform(self,y_test,categories_test):
		interpreter = dict()
		try:
			for i in range(len(categories_test)):
				c = categories_test[i]
				j = self._categories.index(c)
				interpreter[i] = j
		except ValueError :
			print ('y_test_transform failed')

		y_test_transformed = list()
		y_test_transformed = [interpreter[index] for index in y_test]
		return y_test_transformed

	def y_test_reindexing(self, data_test):
		y_test = data_test.target
		y_test = self.__y_test_transform(y_test,data_test.target_names)
		return np.asarray(y_test)
	# --------------------------------------------------------------------------------------------

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

		score = metrics.accuracy_score(y_test, pred) # 模型的评价accuracy_score/recall_score
		print("accuracy_score:   %0.3f" % score)
		score = metrics.recall_score(y_test, pred) # 模型的评价accuracy_score/recall_score
		print("recall_score:   %0.3f" % score)
		score = metrics.f1_score(y_test, pred) # 模型的评价accuracy_score/recall_score
		print("f1-score:   %0.3f" % score)

		if hasattr(self._clf, 'coef_'):
			print("dimensionality: %d" % self._clf.coef_.shape[1])
			print("density: %f" % density(self._clf.coef_))
			if print_top10 and feature_names is not None:
				print("top 10 keywords per class:")
				for i, category in enumerate(categories):
					# import pdb;pdb.set_trace()
					try:
						top10 = np.argsort(self._clf.coef_[i])[-10:]					
						print("%s: %s"% (category, " ".join(feature_names[top10])))
					except Exception, e:
						continue
			print()

		if print_report:
			print("classification report:")
			print(metrics.classification_report(y_test, pred))
			print(metrics.classification_report(y_test, pred,target_names=categories))

		if print_cm:
			print("confusion matrix:")
			print(metrics.confusion_matrix(y_test, pred))

		print()

	@staticmethod
	def measure(y_test, pred):
		""" 静态方法，评估预测的效果,不关心分类器内部的细节 """
		accuracy_score = metrics.accuracy_score(y_test, pred) # 模型的评价accuracy_score/recall_score
		print("accuracy_score:   %0.3f" % accuracy_score)
		
		try:
			recall_score = metrics.recall_score(y_test, pred) # 模型的评价accuracy_score/recall_score
			f1_score = metrics.f1_score(y_test, pred) # 模型的评价accuracy_score/recall_score
		except Exception, e:
			# *** ValueError: pos_label=1 is not a valid label: array([u'2503', u'2511'], dtype='<U4')
			recall_score = np.nan
			f1_score = np.nan
		print("recall_score:   %0.3f" % recall_score)
		print("f1-score:   %0.3f" % f1_score)

		print("classification report:")
		classification_report = metrics.classification_report(y_test, pred)
		print(classification_report)

		print("confusion matrix:")
		confusion_matrix = metrics.confusion_matrix(y_test, pred)
		# confusion_matrix过大时返回的结果看不到内部
		np.set_printoptions(threshold='nan')
		print( confusion_matrix )

		return accuracy_score,recall_score,f1_score,classification_report,confusion_matrix