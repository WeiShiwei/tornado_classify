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
###############################################################################
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '..'))

from base_clf import BaseClf
from feature_extraction.text import TfidfVectorizer
# from sklearn_classifier import LinearSVC
from sklearn import linear_model
from sklearn.externals import joblib

# from sklearn.feature_extraction import DictVectorizer
# from sklearn.feature_extraction.text import CountVectorizer

from _stop_words import stop_words

class SGDClassifier(BaseClf):
	"""docstring for SGDClassifier"""


	def __init__(self, clf=linear_model.SGDClassifier(penalty='l2',loss='log',class_weight=None), 
		vtr=TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=stop_words,encoding='utf-8')):#,token_pattern=r'\b\w+\b'
		super(SGDClassifier, self).__init__(clf, vtr)
	
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


from datasets import load_files
from sklearn.externals import joblib
import os
import global_variables as gv

from sklearn import cross_validation
from sklearn import svm
from sklearn.utils import shuffle

from feature_selection import chi2

def main():

	sgd_clf = SGDClassifier()
	# -------------------------------------------------------------------------------------------------
	multilevel_code = '01'
	# categories_select = ['5001','5003','5005']
	categories_select = None
	data_train = load_files.fetch_Bunch_datas(multilevel_code,categories=categories_select)
	# -------------------------------------------------------------------------------------------------
	X_train = sgd_clf.transform(data_train)
	y_train = data_train.target
	# X_train = chi2.fit_transform(X_train, y_train)

	X, y = shuffle(X_train, y_train, random_state=13)
	offset = int(X.shape[0] * 0.6)
	X_train, y_train = X[:offset], y[:offset]
	X_test, y_test = X[offset:], y[offset:]

	sgd_clf.fit(X_train, y_train)
	pred = sgd_clf._clf.predict(X_test)
	sgd_clf.score(y_test,pred)
	# import pdb;pdb.set_trace()
	proba = sgd_clf.predict_proba(X_test)
# -------------------------------------------------------------------------------------------------
	
	import pdb;pdb.set_trace()
	file_name = os.path.join(gv.models_path,multilevel_code)+'/'+'_'.join(data_train.target_names)+'.joblib'
	sgd_clf.dump(file_name)
	

if __name__ == "__main__":
	main()