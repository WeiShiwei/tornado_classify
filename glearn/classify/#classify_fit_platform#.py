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
	tempdir = tempfile.mkdtemp()
	try:
		# potential problems(2014/12/23)
		for (branch_plus, hierarchical_categories) in task.task_data.items():
			for hc in hierarchical_categories:
				src = os.path.join( task.task_data_source_path, hc )
				dst = os.path.join( tempdir, branch_plus )
				if not os.path.exists(dst):
					os.makedirs(dst)

				commandLine = 'cp '
				commandLine += src+'/* '
				commandLine += dst
				(status, output)=commands.getstatusoutput(commandLine)
		data_train = load_files(tempdir, encoding=gv.encoding, load_files_type=task.load_files_type) 
		print '\nhierarchical branch:{0} \nhierarchical categoris:{1}'.format( task.task_name, task.task_data)
		
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
		# print file_name
		# print ''
		print '{0}\n'.format( file_name )

		return HierarchicalFitClf( task.identity, task.task_name, linear_svc_clf )
	except Exception, e:
		print e
		return None
	finally:
		shutil.rmtree(tempdir)

class RedirectSysStdOut(object):
	"""docstring for RedirectSysOut
	try 'Defining a Decorator That Takes Arguments'
	"""
	redirect = True
	
	@classmethod
	def redirect_string_io(self):
		RedirectSysStdOut.infoStrIO = StringIO.StringIO()
		RedirectSysStdOut.saveout = sys.stdout
		if RedirectSysStdOut.redirect:
			sys.stdout = RedirectSysStdOut.infoStrIO ###
		else:
			pass

	@classmethod
	def redirect_undo(self):
		sys.stdout = RedirectSysStdOut.saveout
		fit_info = RedirectSysStdOut.infoStrIO.getvalue()
		RedirectSysStdOut.infoStrIO.close()

		return fit_info

def string_io(func):
	""" Decorator that log learn infomation"""
	@wraps(func)
	def wrapper(*args, **kwargs):
		RedirectSysStdOut.redirect_string_io() # 重定向标准输出
		result = func(*args, **kwargs)
		
		fit_info = RedirectSysStdOut.redirect_undo()
		print fit_info
		
		return fit_info,result
	return wrapper

class ClassifyFitTask(object):
	""" plan B """
	def __init__(self, task_name, task_data, task_data_source_path = None, identity=None):
		super(ClassifyFitTask, self).__init__()
		self.task_name = task_name
		self.task_data = task_data
		self.task_data_source_path = task_data_source_path
		self.identity = identity

	def run(self):
		tempdir = tempfile.mkdtemp()
		try:
			for (branch_plus, hierarchical_categories) in self.task_data.items():
				for hc in hierarchical_categories:
					src = os.path.join( self.task_data_source_path, hc )
					dst = os.path.join( tempdir, branch_plus )
					if not os.path.exists(dst):
						os.makedirs(dst)

					commandLine = 'cp '
					commandLine += src+'/* '
					commandLine += dst
					(status, output)=commands.getstatusoutput(commandLine)
			data_train = load_files(tempdir, encoding=gv.encoding) 
			print 
			print 'hierarchical branch:',self.task_name
			print 'hierarchical categoris:',self.task_data
			# ---------------------------linear_svc训练模型
			linear_svc_clf = LinearSVCClassifier()
			X_train = linear_svc_clf.transform(data_train)
			y_train = data_train.target

			t0 = time()
			sample_weight = _balance_weights(y_train)
			linear_svc_clf.fit(X_train, y_train)
			print("done in %fs" % (time() - t0))

			# ---------------------------序列化存储
			goal = os.path.join( gv.get_model_home(), self.identity, self.task_name)
			if not os.path.exists(goal):
			    os.makedirs(goal)
			file_name = os.path.join( goal, ''.join( ['hierarchical|',self.identity,'@',self.task_name,'@','LinearSVCClassifier.joblib'] ) )
			linear_svc_clf.dump(file_name)
			print file_name
			print ''

			return HierarchicalFitClf( self.identity, self.task_name, linear_svc_clf )
		except Exception, e:
			print e
			return None
		finally:
			shutil.rmtree(tempdir)


class ClassifyFit(object):
	"""docstring for ClassifyFit"""
	
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
	def flat_classify_fit(self, identity, archive_file_name ,load_files_type):
		""" 扁平文本分类 """
		target_dir,train_folder = ClassifyFit.__uncompress_common(identity, archive_file_name)
		ClassifyFit.__rmtree_common( identity )
		
		# 加载训练数据并且压缩/序列化存储
		data_train=load_files(train_folder, encoding=gv.encoding ,load_files_type=load_files_type)
		# compressed_content = codecs.encode(pickle.dumps(data_train), 'zlib_codec')
		# cache_path = os.path.join(data_home, archive_name+'.pkl')
		# open(cache_path, 'wb').write(compressed_content)
		# shutil.rmtree(target_dir)

		# 训练linearSVC模型
		linear_svc_clf = LinearSVCClassifier()
		X_train = linear_svc_clf.transform(data_train)
		y_train = data_train.target
		t0 = time()
		sample_weight = _balance_weights(y_train)
		linear_svc_clf.fit(X_train, y_train)
		print("done in %fs" % (time() - t0))

		# 序列化存储linearSVC模型
		goal = os.path.join( gv.get_model_home(), identity)
		if not os.path.exists(goal):
		    os.mkdir(goal)
		file_name = os.path.join( goal, ''.join( ['flat|',identity,'@','','@','LinearSVCClassifier.joblib'] ) )
		linear_svc_clf.dump(file_name)
		print '{0}\n'.format( file_name )
		
		return FlatFitClf(identity,linear_svc_clf)

	@classmethod
	def __fetch_hierarchical_category(self, root, single_level=True):
		""" __fetch_hierarchical_category """
		hierarchical_category = dict()
		for path, subdirs, files in os.walk(root):
			if single_level:
				break

		hierarchical_category_list = subdirs
		# 20news-bydate example:
		# ['sci.med', 'alt.atheism', 'talk.politics.mideast', 'soc.religion.christian', 'sci.crypt', 
		# 	'rec.sport.baseball', 'comp.sys.ibm.pc.hardware', 'talk.politics.guns', 'sci.electronics', 
		# 	'comp.graphics', 'comp.windows.x', 'rec.autos', 'talk.politics.misc', 'rec.motorcycles', 
		# 	'talk.religion.misc', 'misc.forsale', 'comp.sys.mac.hardware', 'rec.sport.hockey', 'comp.os.ms-windows.misc', 'sci.space']
		print 'Hierarchical Category List:'
		pprint( sorted(hierarchical_category_list) )
		print 
		return hierarchical_category_list

	@classmethod
	def hierarchical_classify_fit(self, identity, archive_file_name ,load_files_type):
		""" 层次文本分类 """
		target_dir, train_folder = ClassifyFit.__uncompress_common(identity, archive_file_name)
		ClassifyFit.__rmtree_common( identity )

		hierarchical_category_list = ClassifyFit.__fetch_hierarchical_category(train_folder)

		tree_branch_route_set = set()
		for hierarchical_category in hierarchical_category_list:
			category_list = hierarchical_category.split('.')
			for i in xrange( len(category_list) ):
				tree_branch_route_set.add( '.'.join(category_list[0:i+1]) )
		
		print 'Hierarchical Branch Set:'
		pprint(sorted( tree_branch_route_set ))
		print 
		# 20news-bydate example:
		# ['alt', 
		#  'alt.atheism', 
		#  'comp', 
		#  'comp.graphics', 
		#  'comp.os', 
		#  'comp.os.ms-windows', 
		#  'comp.os.ms-windows.misc', ...]
		
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
		# import pdb;pdb.set_trace()
		print tree_branch_route_dict
		# (Pdb) tree_branch_route_dict['comp']
		# defaultdict(<type 'list'>, {
		# 	'comp.sys': ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware'], 
		# 	'comp.os': ['comp.os.ms-windows.misc'], 
		# 	'comp.windows': ['comp.windows.x'], 
		# 	'comp.graphics': ['comp.graphics']})

		hierarchical_fit_clfs = list()


		# import pdb;pdb.set_trace()
		# -*- Plan A -*-
		TASKS = [ TASK( key, value, 
						task_data_source_path=train_folder, 
						identity=identity,
						load_files_type=load_files_type ) 
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

		# -*- Plan B -*-
		# clf_fit_tasks = [ ClassifyFitTask( key, 
		# 									value, 
		# 									task_data_source_path=train_folder, 
		# 									identity=identity ) 
		# 				for (key,value) in tree_branch_route_dict.items()

		# hierarchical_fit_clfs = list()
		# for task in clf_fit_tasks:
		# 	hierarchical_fit_clfs.append( task.run() )			
		
		return hierarchical_fit_clfs

	@classmethod
	def learn(self, identity, archive_file_name, load_files_type = 'document', model_learning_type='flat' ): 
		""""""
		fit_sys_stdout, clfs = None, None
		if model_learning_type == 'flat':
			clfs = ClassifyFit.flat_classify_fit( identity, archive_file_name ,load_files_type)
			return fit_sys_stdout, clfs
		
		clfs = ClassifyFit.hierarchical_classify_fit( identity, archive_file_name ,load_files_type)
		return clfs

def main():
	identity = 'gldjc'
	archive_file_name = "gldjc-train.tar.gz"
	fit_sys_stdout,clfs = ClassifyFit.learn( identity, 
										archive_file_name, 
										load_files_type = 'line', 
										model_learning_type = 'hierarchical' )


if __name__ == "__main__":
	main()