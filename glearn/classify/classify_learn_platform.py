#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

import pickle
import shutil
import codecs
import tarfile
import zipfile
import glob
import tempfile
import fnmatch

import commands
import StringIO
from time import time
from pprint import pprint
from functools import wraps
from collections import defaultdict, namedtuple
from multiprocessing import Pool

import global_variables as gv
from _weights import _balance_weights
from linear_svc import LinearSVCClassifier

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../../'))
from glearn.datasets.load_files_platform import load_files


HierarchicalFitClf = namedtuple("HierarchicalFitClf", "identity branch clf")
FlatFitClf = namedtuple("FlatFitClf", "identity clf")
TASK = namedtuple("TASK", "task_name, task_data, task_data_source_path, identity, load_files_type")

def learn( *args ):
	""" Pool( processes=gv.PROCESSES ).apply_async(learn, t) for t in TASKS """
	task = TASK(*args)

	data_home = task.task_data_source_path
	multilevel_codes = task.task_name.split('.')
	data_train = load_files(data_home,multilevel_codes=multilevel_codes,
							encoding='utf-8',feature_engineering = True,
							load_files_type='line')
	print '\nhierarchical branch:{0} \nhierarchical categoris:{1}'.format( task.task_name, data_train.target_names)
	
	try:
		# linear_svc训练模型
		linear_svc_clf = LinearSVCClassifier()
		X_train = linear_svc_clf.transform(data_train)
		y_train = data_train.target
		t0 = time()
		sample_weight = _balance_weights(y_train)
		linear_svc_clf.fit(X_train, y_train)
		print("done in %fs" % (time() - t0))

		# 序列化存储linearSVC模型
		goal = os.path.join( gv.get_model_home(), task.identity, task.task_name)
		if not os.path.exists(goal):
		    os.makedirs(goal)
		file_name = os.path.join( goal, ''.join( ['hierarchical|',task.identity,'@',task.task_name,'@','LinearSVCClassifier.joblib'] ) )
		linear_svc_clf.dump(file_name)
		print '{0}\n'.format( file_name )
		return HierarchicalFitClf( task.identity, task.task_name, linear_svc_clf )
	except Exception, e:
		return None
	
class ClassifyLearn(object):
	"""docstring for ClassifyLearn"""
	
	@classmethod
	def __uncompress_common(self, identity, archive_file_name):
		""" __uncompress_common """
		data_home = os.path.join( gv.get_data_home(), identity)
		archive_path=os.path.join(data_home, archive_file_name)
		try:
			if archive_file_name.endswith('.zip'):
			    archive_name = archive_file_name.rstrip('.zip') 
			    target_dir = os.path.join( data_home, archive_name)
			    zipfile.ZipFile(archive_path).extractall(path=target_dir)
			elif archive_file_name.endswith('.tar.gz'):
			    archive_name = archive_file_name.rstrip('.tar.gz') 
			    target_dir = os.path.join(data_home, archive_name)
			    tarfile.open(archive_path, "r:gz").extractall(path=target_dir)
			elif os.path.isdir( archive_path ): # for gldjc
				target_dir = archive_path
			else:
				print 'archive_file_name is not .zip & .tar.gz & directory'
				return None,None

			train_folder = glob.glob( os.path.join(target_dir,'*train'))[0]
			#\ test_folder = glob.glob( os.path.join(target_dir,'*test'))[0]
		except Exception, e:
		    raise e
		return target_dir,train_folder
	
	@classmethod
	def __rmtree_common(self, identity):
		target_dir = os.path.join( gv.get_model_home(), identity)
		if os.path.exists( target_dir ):
			shutil.rmtree( target_dir )

	@classmethod
	def _fetch_hierarchical_category(self, root, single_level=True):
		""" __fetch_hierarchical_category """
		root_backup = root
		hierarchical_path_set = set()

		hierarchical_category = dict()
		for path, subdirs, files in os.walk(root):
			for name in files:
				if fnmatch.fnmatch(name,'*.train'):
					hierarchical_path_set.add( path )

		hierarchical_categorys = list()
		for hierarchical_path in hierarchical_path_set:
			hierarchical_categorys.append( '.'.join( 
				filter(lambda x:x is not '', hierarchical_path.split('/')[len(root.split('/')):] ) 
			))

		print 'Hierarchical Category List:',sorted(hierarchical_categorys)
		return hierarchical_categorys

	@classmethod
	def hierarchical_classify_fit(self, identity, archive_file_name ,load_files_type):
		""" 层次文本分类 """
		target_dir, train_folder = ClassifyLearn.__uncompress_common(identity, archive_file_name)
		ClassifyLearn.__rmtree_common( identity )

		# 
		hierarchical_category_list = ClassifyLearn._fetch_hierarchical_category(train_folder)
		# -----------------------------Not required
		tree_branch_route_set = set()
		for hierarchical_category in hierarchical_category_list:
			category_list = hierarchical_category.split('.')
			for i in xrange( len(category_list) ):
				tree_branch_route_set.add( '.'.join(category_list[0:i+1]) )		

		# 层次目录体系如果看成一棵树的话（根节点为''），
		# tree_branch_route_set存储从根节点到其他节点（不仅仅是叶节点）的路径
		print 'Hierarchical Branch Set:',sorted( tree_branch_route_set)
		
		tree_branch_route_set.add('')
		tree_branch_route_dict = defaultdict(dict)
		for branch in sorted( tree_branch_route_set ):
			branch_distance = len( branch.split('.') )
			if branch:
				branch_distance_plusOne = branch_distance + 1
				hierarchical_categories = filter(lambda x:x.startswith( branch+'.' ), hierarchical_category_list)
			else:
				branch_distance_plusOne = branch_distance
				hierarchical_categories = filter(lambda x:x.startswith( '' ), hierarchical_category_list)
			
			branchPlus_hierarchicalCategory_dict = defaultdict(list)
			for hierarchical_category in hierarchical_categories:
				branchPlus_hierarchicalCategory_dict[ '.'.join( hierarchical_category.split('.')[:branch_distance_plusOne] )].append( hierarchical_category )

			tree_branch_route_dict[ branch ] = branchPlus_hierarchicalCategory_dict 

		# tree_branch_route_dict字典的键存储根节点到其他节点（不仅仅是叶节点）的路径path
		# 字典的值存储以path为前缀的子路径
		print "tree_branch_route_dict: ",tree_branch_route_dict
		# -----------------------------

		hierarchical_fit_clfs = list()
		# -*- Plan A -*-
		TASKS = [ TASK( key, value, task_data_source_path=train_folder, 
						identity=identity,load_files_type=load_files_type ) 
				for (key,value) in tree_branch_route_dict.items()
		]

		pool = Pool( processes=gv.PROCESSES )
		results = [pool.apply_async(learn, t) for t in TASKS]
		pool.close()
		pool.join()
		
		print 'Ordered results using pool.apply_async():'
		for r in results:
			model = r.get()
			if model is None:
				continue
			print '\t', model
			hierarchical_fit_clfs.append( model )		
		
		return hierarchical_fit_clfs

	@classmethod
	def learn(self, identity, archive_file_name, load_files_type = 'document'): 
		""""""
		clfs = ClassifyLearn.hierarchical_classify_fit( identity, archive_file_name ,load_files_type)
		return clfs

def main():
	identity = 'gldjc'
	archive_file_name = "gldjc-train.tar.gz"
	clfs = ClassifyLearn.learn( identity, archive_file_name, load_files_type = 'line')


if __name__ == "__main__":
	main()