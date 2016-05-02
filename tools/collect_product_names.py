#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import fnmatch

MAIN_USER = 'weishiwei'
root = os.path.join('/home',MAIN_USER,'scikit_learn_data')
target_dir = os.path.join('/home',MAIN_USER,'product_names')

def collect(file_path):
	product_name_list = list()
	with open(file_path) as infile:
		lines = infile.readlines()
		for line in lines:
			line = line.strip()
			if line == '':
				continue
			product_name = line.split()[0]
			product_name_list.append(product_name)
	product_name_set = set(product_name_list)

	print file_path
	with open(os.path.join(target_dir,os.path.basename(file_path)),'wb') as outfile:
		outfile.write( '\n'.join(product_name_set) )
	# return product_name_set

def main():
	patterns='*.txt';single_level=False;yield_folders=False
	patterns = patterns.split(';')

	for path, subdirs, files in os.walk(root):
		if yield_folders:
			files.extend(subdirs)
		files.sort()
		for name in files:
			for pattern in patterns:
				if fnmatch.fnmatch(name,pattern):
					txt_file = os.path.join(path,name)
					print collect(txt_file)
					# import pdb;pdb.set_trace()
		if single_level:
			break




if __name__ == '__main__':
	main()