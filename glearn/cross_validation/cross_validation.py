#!/usr/bin/python
# coding=utf-8

import numpy as np
from sklearn import cross_validation
# from sklearn import datasets
# from sklearn import svm



def cross_val_score(clf, data, target, cv, scoring):
	scores = cross_validation.cross_val_score(clf, data, target, cv=5, scoring=scoring)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	return scores