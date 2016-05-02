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
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '..'))
from base_clf import BaseClf
from sklearn_classifier import MultinomialNB
from feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from _stop_words import stop_words

class NaiveBayesClassifier(BaseClf):
	"""docstring for NaiveBayesClassifier"""

	# def __init__(self, clf=MultinomialNB(alpha=.01), 
	# 	vtr = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=stop_words,encoding='utf-8')):
	# 	super(NaiveBayesClassifier, self).__init__(clf, vtr)
	def __init__(self, clf=MultinomialNB(alpha=.01), 
		vtr = CountVectorizer(ngram_range=(1, 2),token_pattern='(?u)\\b\\w\\w+\\b', min_df=1)):
		super(NaiveBayesClassifier, self).__init__(clf, vtr)

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



from datasets import load_files
from sklearn.externals import joblib
import os
import global_variables as gv
from sklearn import cross_validation
from sklearn.utils import shuffle

def main():

	naive_bayes_clf = NaiveBayesClassifier()
	import pdb;pdb.set_trace()
	# -------------------------------------------------------------------------------------------------

	multilevel_code = ''
	categories_select = None
	data_train = load_files.fetch_Bunch_datas(multilevel_code,categories=categories_select)

	X_train = naive_bayes_clf.transform(data_train)
	y_train = data_train.target
	# X_train = chi2.fit_transform(X_train, y_train)

	X, y = shuffle(X_train, y_train, random_state=13)
	offset = int(X.shape[0] * 0.001)
	X_train, y_train = X[:offset], y[:offset]
	X_test, y_test = X[offset:], y[offset:]

	import pdb;pdb.set_trace()
	naive_bayes_clf.fit(X_train, y_train)
	# naive_bayes_clf._clf.partial_fit(X_train, y_train)
	pred = naive_bayes_clf._clf.predict(X_test)
	naive_bayes_clf.score(y_test,pred)
	import pdb;pdb.set_trace()
# ---------------------------------cross_validation----------------------------------------------------------------
	import pdb;pdb.set_trace()
	# clf = svm.SVC(kernel='linear', C=1)
	X_train = naive_bayes_clf.transform(data_train)
	clf = naive_bayes_clf._clf
	scores = cross_validation.cross_val_score( clf, X_train, data_train.target, cv=5)
	scores
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# 	multilevel_code = '25'
# 	categories_select = ['2503','2511']
# 	data_test = load_files.fetch_Bunch_datas(multilevel_code,description="basic_datas_lv1_25",categories=categories_select,is_for_train=False)

# 	pred = naive_bayes_clf.predict(data_test)
# 	predict_res = naive_bayes_clf.predict_categorys(pred)
# # -------------------------------------------------------------------------------------------------
# 	y_test = naive_bayes_clf.y_test_reindexing(data_test)
# 	naive_bayes_clf.score( y_test, pred)

	# y_test = data_test.target
	# naive_bayes_clf.score( y_test, pred)

	import pdb;pdb.set_trace()

if __name__ == "__main__":
	main()
