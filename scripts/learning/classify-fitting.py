#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import commands
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../..'))

from glearn.classify.classify_learn_platform import ClassifyLearn

identity = 'gldjc'

def main():
	"""
	python classify-fitting archiveFilePath
	"""
	print "uasage : python classify-fitting archiveFilePath"
	# print "sys.argv: ",sys.argv
	try:
		archive_file_path = sys.argv[1]
		archive_file_name = os.path.basename(archive_file_path)
	except Exception, e:
		raise e
		sys.exit(0)

	# clear home space
	classify_identity_home = os.path.expanduser( os.path.join('~','classify_learn_data',identity))
	if os.path.exists( classify_identity_home ):
		shutil.rmtree( classify_identity_home )
	os.makedirs( classify_identity_home )

	# copy archive file to classify_identity_home
	commandLine = 'cp '
	commandLine += archive_file_path+' '
	commandLine += classify_identity_home
	print commandLine
	(status, output)=commands.getstatusoutput(commandLine)


	# archive_file_name = "gldjc-data.tar.gz"
	# print archive_file_name
	clfs = ClassifyLearn.learn( identity, archive_file_name, load_files_type = 'line' )

if __name__ == '__main__':
	main()