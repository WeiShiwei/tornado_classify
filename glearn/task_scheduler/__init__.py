# -*- coding: utf-8 -*-
from tasks_classify import gldjc, gldzb

IDENTITY_APP_DICT = {
	'gldjc':gldjc,
	'gldzb':gldzb
}

class TaskScheduler(object):
	"""docstring for TaskScheduler"""
	def __init__(self, arg):
		super(TaskScheduler, self).__init__()
		self.arg = arg

	@classmethod
	def apply_async(self, identity, docs):
		# import pdb;pdb.set_trace()
		try:
			res = IDENTITY_APP_DICT[identity].predict.apply_async( (identity, docs), queue = identity )
		except KeyError, e:
			print e
			res = None
		
		return res
