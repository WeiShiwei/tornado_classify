#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../..'))
from time import time
from sklearn.utils import shuffle
import numpy as np

from glearn.datasets import load_files
from glearn.classify.linear_svc import LinearSVCClassifier
from glearn.classify.libsvm_svc import LibSVMClassifier
from glearn.classify.random_forest import RandomForestClassifier
from glearn.classify.sgd_clf import SGDClassifier
from glearn.classify._weights import _balance_weights
import glearn.classify.global_variables as gv

from glearn.classify.orm import BaseMaterialType

models_path = gv.MODELS_PATH

def LinearSVCClassifier_learn(multilevel_code,categories_select,ratio):
	linear_svc_clf = LinearSVCClassifier()

	multilevel_code = multilevel_code
	categories_select = categories_select
	data_train = load_files.fetch_Bunch_datas(multilevel_code,categories=categories_select)
	# # -------------------fit_transform-------------------------------
	X_train = linear_svc_clf.transform(data_train)
	y_train = data_train.target
	#\ X_train = chi2.fit_transform(X_train, y_train)

	X, y = shuffle(X_train, y_train, random_state=13)
	offset = int(X.shape[0] * ratio)
	X_train, y_train = X[:offset], y[:offset]
	X_test, y_test = X[offset:], y[offset:]

	t0 = time()
	sample_weight = _balance_weights(y_train)
	linear_svc_clf.fit(X_train, y_train)
	print("done in %fs" % (time() - t0))

	# -------------------------------------------------------------------------------------------------
	goal = os.path.join(models_path,multilevel_code)
	if not os.path.exists(goal):
		os.mkdir(goal)
	file_name = os.path.join(models_path,multilevel_code)+'/'+'LinearSVCClassifier@' + '_'.join(data_train.target_names)+'.joblib'
	print file_name
	print 
	linear_svc_clf.dump(file_name)

def main():
	if not os.path.exists(models_path):
		os.mkdir(models_path)	
	all_lv1_codes = sorted(BaseMaterialType.all_lv1_codes())
	all_lv1_codes = [u'01', u'02', u'03', u'04', u'05', 
					u'06', u'07', u'08', u'09', u'10', 
					u'11', u'12', u'13', u'14', u'15', 
					u'16', u'17', u'18', u'19', u'20', 
					u'21', u'22', u'23', u'24', u'25', 
					u'26', u'27', u'28', u'29', u'30', 
					u'31', u'32', u'33', u'50', u'51', 
					u'52', u'53', u'54', u'55', u'56', 
					u'57', u'58', u'80', u'98', u'99']# has no u'00'
	all_lv1_codes.append('')
	

	# import pdb;pdb.set_trace() ###
	# -----------------------------------------------------------------------------
	group_list = all_lv1_codes;print ' '.join(group_list)
	print group_list
	for base_code in group_list:
		multilevel_code = base_code
		categories_select = None
		ratio = 1.0
		LinearSVCClassifier_learn(multilevel_code,categories_select,ratio)
		#\ SGDClassifier_learn(multilevel_code,categories_select,ratio)
	# -----------------------------------------------------------------------------


if __name__ == "__main__":
	main()
