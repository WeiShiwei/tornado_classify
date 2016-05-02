#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: WeiShiwei <weishiwei920@163.com>
# License: BSD 3 clause

from __future__ import print_function

import logging
import scipy
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
# from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble

from _stop_words import stop_words

class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

class RandomForestClassifier(BaseClf):
	"""docstring for RandomForestClassifier
	评价：占用的内存空间太大
	"""

	def __init__(self, clf=ensemble.RandomForestClassifier(n_estimators=12, max_depth=None, min_samples_split=1, random_state=0, n_jobs=12),
		vtr=TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=stop_words,
							ngram_range=(1, 2),token_pattern='(?u)\\b\\w\\w+\\b',
							max_features=5000)): 
		super(RandomForestClassifier, self).__init__(clf, vtr)
	
	def fetch_categorys(self,pred):
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

	def predict(self, docs):
		""" docs既可以是列表格式的文档集，又可以是Bunch格式的文档集
		"""
		vectorizer_ = self.vectorizer_
		if isinstance(docs, list):# 列表格式的文档集
			#\ X_test = vectorizer_.fit_transform(docs) # fit_transform是依据当前的docs重新矢量化转换
			X_test = vectorizer_.transform(docs) # fit_transform是依据vectorizer_的vocabulary_进行矢量化转换
			X_test = X_test.toarray()
		elif isinstance(docs, Bunch):# Bunch格式的文档集
			#\ <class 'datasets.load_files.Bunch'>
			#\ X_test = vectorizer_.fit_transform(docs.data)
			X_test = vectorizer_.transform(docs.data)
			X_test = X_test.toarray()
		elif isinstance(docs, scipy.sparse.csr.csr_matrix) or isinstance(docs, np.ndarray):
			X_test = docs.toarray()
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


import os
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.externals import joblib

from glearn.datasets import load_files as load_files_bunch
from glearn.datasets.load_files_platform import load_files # 训练一级类模型是有问题的
from glearn.model_selection.plot_learning_curve import plot_learning_curve
import global_variables as gv

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '..'))
from feature_extraction.text import TfidfVectorizer
from _weights import _balance_weights

def _plot_learning_curve(multilevel_code=''):
	t0 = time()
	random_forest_clf = RandomForestClassifier()
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
	X_train = random_forest_clf.transform(data_train)
	X_train = X_train.astype(np.float32) ###
	y_train = data_train.target

	# plot_learning_curve #
	X, y = shuffle(X_train, y_train, random_state=13)
	title = "Learning Curves (RandomForestClassifier)"
	# Cross validation with 100 iterations to get smoother mean test and train
	# score curves, each time with 20% data randomly selected as a validation set.
	cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10,
										test_size=0.2, random_state=0)
	estimator = random_forest_clf._clf
	plt = plot_learning_curve(estimator, title, X.toarray(), y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
	# plt.show()
	
	return plt

def main():
	random_forest_clf = RandomForestClassifier()
	# -------------------------------------------------------
	multilevel_code = '04'
	categories_select = None
	# categories_select =  ['28', '54', '98']
	# categories_select =  ['00', '04', '05', '12', '30', '31', '32', '33', '52', '56', '80', '99']
	# categories_select =  ['01', '02', '03', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '26', '27', '29', '50', '51', '53', '57', '58']
	# categories_select =  ['25', '55']
	data_train = load_files.fetch_Bunch_datas(multilevel_code, categories = categories_select)


	#data_train = load_files.fetch_fine_grained_level_datas('', categories = categories_select)

	# -------------------------------------------------------------------------------------------------
	# import pdb;pdb.set_trace()
	X_train = random_forest_clf.transform(data_train)
	X_train = X_train.astype(np.float32)
	y_train = data_train.target

	X, y = shuffle(X_train, y_train, random_state=13)
	offset = int(X.shape[0] * 0.8)
	X_train, y_train = X[:offset], y[:offset]
	X_test, y_test = X[offset:], y[offset:]

	sample_weight = _balance_weights(y_train)
	# 至少这里函数给提供了面对样本不均衡问题的主动的办法
	# random_forest_clf.fit(X_train.toarray(), y_train, sample_weight)
	random_forest_clf.fit(X_train.toarray(), y_train)
	pred = random_forest_clf.predict(X_test)
	random_forest_clf.score(y_test,pred)
	
	# proba = random_forest_clf.predict_proba(X_test)
	# print proba
	import pdb;pdb.set_trace()
	# -------------------------------------------------------------------------------------------------
	
	

if __name__ == "__main__":
	# main()

	print("usage: python random_forest.py 01 | python random_forest.py")
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

