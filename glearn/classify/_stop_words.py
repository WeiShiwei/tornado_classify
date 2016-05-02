#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import global_variables as gv
reload(sys)                         #  
sys.setdefaultencoding('utf-8')     # 

stop_words = list()
with open(gv.stop_words_path,'r') as infile:
	# stop_words = infile.readlines()
	for word in infile.readlines():
		stop_words.append(unicode(word.strip('\r\n')))