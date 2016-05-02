#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../../..'))
from time import time
from sklearn.utils import shuffle

from glearn.datasets import load_files
from glearn.classify.linear_svc import LinearSVCClassifier
from glearn.classify.libsvm_svc import LibSVMClassifier
from glearn.classify.random_forest import RandomForestClassifier
from glearn.classify.sgd_clf import SGDClassifier
import glearn.classify.global_variables as gv
from glearn.classify._weights import _balance_weights

def SGDClassifier_fit(multilevel_code,categories_select,ratio):
	sgd_clf = SGDClassifier()

	multilevel_code = multilevel_code
	categories_select = categories_select
	data_train = load_files.fetch_Bunch_datas(multilevel_code,categories=categories_select)
	# # -------------------fit_transform-------------------------------
	X_train = sgd_clf.transform(data_train)
	y_train = data_train.target
	#\ X_train = chi2.fit_transform(X_train, y_train)

	X, y = shuffle(X_train, y_train, random_state=13)
	offset = int(X.shape[0] * ratio)
	X_train, y_train = X[:offset], y[:offset]
	X_test, y_test = X[offset:], y[offset:]

	t0 = time()
	sample_weight = _balance_weights(y_train)
	sgd_clf.fit(X_train, y_train)
	print("done in %fs" % (time() - t0))

	# -------------------------------------------------------------------------------------------------
	goal = os.path.join(gv.models_path,multilevel_code)
	if not os.path.exists(goal):
		os.mkdir(goal)
	file_name = os.path.join(gv.models_path,multilevel_code)+'/'+'SGDClassifier@' + '_'.join(data_train.target_names)+'.joblib'
	print file_name
	sgd_clf.dump(file_name)


def main():
	if not os.path.exists(gv.models_path):
		os.mkdir(gv.models_path)
	all_lv1_codes = sorted(BaseMaterialType.all_lv1_codes())

	group_list = all_lv1_codes;print ' '.join(group_list)
	for base_code in group_list:
		multilevel_code = base_code
		categories_select = None
		ratio = 1.0

		SGDClassifier_fit(multilevel_code,categories_select,ratio)
		

if __name__ == "__main__":
	main()