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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

class GradientBoostingClassifier(BaseClf):
	"""docstring for GradientBoostingClassifier
	梯度提升树
	"""
	def __init__(self, clf = GradientBoostingClassifier(n_estimators=100, 
									learning_rate=0.1,
									subsample=0.5,
									# max_depth=1, 
									random_state=0),
				vtr=TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')):
		super(GradientBoostingClassifier, self).__init__(clf, vtr)
	# def __init__(self, clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,max_depth=1, random_state=0), 
	# 	vtr=TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')):
	# 	super(GradientBoostingClassifier, self).__init__(clf, vtr)
	
	# def predict_categorys(self,pred):
	# 	"""AttributeError: 'LinearSVC' object has no attribute 'predict_proba'
	# 	"""
	# 	return [self._categories[cate_index] for cate_index in pred]

	# def predict_category(self,pred):
	# 	"""AttributeError: 'LinearSVC' object has no attribute 'predict_proba'
	# 	"""
	# 	return self._categories[pred]

	def serialization_store(self,file_name):
		# file_name = os.path.join(gv.models_path,multilevel_code)+'/'+'_'.join(data_train.target_names)+'.joblib'
		joblib.dump(self, file_name)

	@classmethod
	def serialization_load(self,file_name):
		return joblib.load(file_name)





from sklearn import cross_validation
from datasets import load_files

import os
import global_variables as gv

from sklearn import ensemble
from sklearn.utils import shuffle

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '..'))
from feature_extraction.text import TfidfVectorizer
from feature_selection import chi2
def main():
	"""
	Traceback (most recent call last):
  File "gradient_boosting.py", line 139, in <module>
    main()
  File "gradient_boosting.py", line 102, in main
    X_train, y_train = X[:offset].toarray(), y[:offset]
  File "/usr/lib/python2.7/dist-packages/scipy/sparse/compressed.py", line 790, in toarray
    return self.tocoo(copy=False).toarray(order=order, out=out)
  File "/usr/lib/python2.7/dist-packages/scipy/sparse/coo.py", line 239, in toarray
    B = self._process_toarray_args(order, out)
  File "/usr/lib/python2.7/dist-packages/scipy/sparse/base.py", line 699, in _process_toarray_args
    return np.zeros(self.shape, dtype=self.dtype, order=order)
ValueError: array is too big.
<---------------------X_train = chi2.fit_transform(X_train, y_train)

	Traceback (most recent call last):
  File "gradient_boosting.py", line 139, in <module>
    main()
  File "gradient_boosting.py", line 113, in main
    gradient_boosting_clf.fit(X_train, y_train)
  File "/home/weishiwei/800w_classifier/tornado_classify/glearn/classify/base_clf.py", line 89, in fit
    self._clf.fit(X_train,y_train)
  File "/usr/local/lib/python2.7/dist-packages/sklearn/ensemble/gradient_boosting.py", line 890, in fit
    return super(GradientBoostingClassifier, self).fit(X, y)
  File "/usr/local/lib/python2.7/dist-packages/sklearn/ensemble/gradient_boosting.py", line 554, in fit
    check_ccontiguous=True)
  File "/usr/local/lib/python2.7/dist-packages/sklearn/utils/validation.py", line 230, in check_arrays
    array = np.ascontiguousarray(array, dtype=dtype)
  File "/usr/lib/python2.7/dist-packages/numpy/core/numeric.py", line 548, in ascontiguousarray
    return array(a, dtype, copy=False, order='C', ndmin=1)
MemoryError
<---------------------X_train = X_train.astype(np.float32)
	"""
	gradient_boosting_clf = GradientBoostingClassifier()
	# -------------------------------------------------------
	multilevel_code = '05'#25
	# categories_select = ['2501','2503','2505','2511','2521','2523','2525','2527','2529','2531','2541','2543']
	# categories_select = ['2501','2505','2521','2523','2525','2527','2529','2531','2541','2543']
	categories_select = None
	# categories_select = ['2503','2511']
	# categories_select = ['2501','2505','2521','2523','2525','2527']
	data_train = load_files.fetch_Bunch_datas(multilevel_code,categories=categories_select)

	# X, y = shuffle(data_train.data, data_train.target, random_state=13)
	# offset = int(X.shape[0] * 0.01)
	# X_train, y_train = X[:offset], y[:offset]
	# X_test, y_test = X[offset:], y[offset:]

	# import pdb;pdb.set_trace()

	X_train = gradient_boosting_clf.transform(data_train)
	X_train = X_train.astype(np.float32)
	y_train = data_train.target
	# import pdb;pdb.set_trace() ###
	# X_train = chi2.fit_transform(X_train, y_train)
	
	X, y = shuffle(X_train, y_train, random_state=13)
	offset = int(X.shape[0] * 0.5)
	X_train, y_train = X[:offset].toarray(), y[:offset]
	X_test, y_test = X[offset:2*offset].toarray(), y[offset:2*offset]

	# X_train, y_train = X[:offset], y[:offset]
	# X_test, y_test = X[offset:], y[offset:]


	# clf_ = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
	# clf_.fit(X_train, y_train)
	# print (clf_.score(X_test, y_test))
	t0 = time()
	gradient_boosting_clf.fit(X_train, y_train)
	print("done in %fs" % (time() - t0))
	print (gradient_boosting_clf._clf.score(X_test, y_test))
	import pdb;pdb.set_trace()
	# clf.fit_transform(data_train)

	#<class 'scipy.sparse.csr.csr_matrix'>
	#<class 'numpy.matrixlib.defmatrix.matrix'>
	#<type 'numpy.ndarray'>
	# X_train = np.asarray(X_train.todense())
	# clf_ = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
	# clf_.fit(X_train.toarray(), y_train)
	# # *** TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
	

	# X_test = clf.transform(data_train)
	# X_test = X_test.toarray()
	# y_test = data_train.target
	# print (clf_.score(X_test, y_test))
	# -------------------------------------------------------
	# file_name = os.path.join(gv.models_path,multilevel_code)+'/'+'_'.join(data_train.target_names)+'.joblib'
	# gradient_boosting_clf.serialization_store(file_name)



if __name__ == "__main__":
	main()