#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import ujson
import fnmatch
import collections
import multiprocessing
from time import time
from collections import defaultdict

import orm
from sklearn.externals import joblib

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../../'))
from glearn.classify import global_variables as gv

class AsyncJobWorker(object):
	"""docstring for AsyncJobWorker"""

	@classmethod
	def __load_clf_joblib(self, identity, root, patterns='*.joblib', 
						single_level=False, yield_folders=False):
		print "="*40
		print "Classifiers Loading Joblib:"
		t0 = time()

		clf_pool = defaultdict(dict)
		clf_context = dict()

		patterns = patterns.split(';')
		for path, subdirs, files in os.walk(root):
			if yield_folders:
				files.extend(subdirs)
			files.sort()
			for name in files:
				for pattern in patterns:
					if fnmatch.fnmatch(name,pattern):
						model_file = os.path.join(path, name)
						identity_,hierarchical_category = name.split('|')[1].split('@')[:2]
						if identity_ == identity:
							clf_context[hierarchical_category] = joblib.load(model_file)
						break
			if single_level:
				break
		print "done in %fs" % (time() - t0)
		return clf_context

	@classmethod
	def _reduce(self, zip_map):
		hierarchicalCategory_docs = defaultdict(list)
		hierarchicalCategory_idxs = defaultdict(list)
		for idx,elem in enumerate(zip_map):
			key,value = elem
			hierarchicalCategory_docs[key].append(value)
			hierarchicalCategory_idxs[key].append(idx)

		return hierarchicalCategory_docs,hierarchicalCategory_idxs


	@classmethod
	def classify(self, identity, docs):
		try:
			clf_context = AsyncJobWorker.clf_context
		except AttributeError, e:
			AsyncJobWorker.clf_context = AsyncJobWorker.__load_clf_joblib( identity, gv.get_model_home())
			clf_context = AsyncJobWorker.clf_context

		initial_classify_res = ['']*len(docs) # 
		intermediate_classify_res = ultimate_classify_res = initial_classify_res
		hierarchicalCategory_docs, hierarchicalCategory_idxs = \
			AsyncJobWorker._reduce(zip(initial_classify_res, docs))

		for k in range(3): # 默认三层
			for hierarchical_category,docs in hierarchicalCategory_docs.items():
				clf = clf_context.get(hierarchical_category,None)
				if not clf:
					continue

				classify_res = clf.fetch_categorys( clf.predict(docs) )
				# intermediate_classify_res update
				for i,idx in enumerate(hierarchicalCategory_idxs[hierarchical_category]):
					intermediate_classify_res[idx] = classify_res[i]
				# print intermediate_classify_res
			
			hierarchicalCategory_docs, hierarchicalCategory_idxs = \
			AsyncJobWorker._reduce(zip(intermediate_classify_res, docs))

		hierarchical_classify_res = intermediate_classify_res
		print "hierarchical_classify_res:",hierarchical_classify_res
		return hierarchical_classify_res

def main():
	identity = 'gldjc'
	docs =['III级螺纹钢   Φ18mm  HRB400 品种:螺纹钢筋;牌号:HRB400;直径Φ(mm):18',
	'弹簧钢 丝 品种 弹簧钢 丝 直径 Φ MM',
	'预应力 钢丝 WLR GB T5223 参数 光面 钢丝 螺旋 肋钢']
	AsyncJobWorker.classify(identity,docs)

if __name__ == '__main__':
	main()