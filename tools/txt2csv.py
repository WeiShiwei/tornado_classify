import os
import sys
import datetime
import glob
import csv

idx = 1
csv_file_path = './names.csv'
writer = csv.writer(file(csv_file_path, 'w'), quoting=csv.QUOTE_ALL)

txt_files = glob.glob('./*txt')
for txt_file in txt_files:
	print txt_file
	lv1_code,lv2_code = os.path.basename(txt_file).rstrip('.txt').split('.')
	names = list()
	with open(txt_file,'rb') as infile:
		names = [line.strip() for line in infile.readlines()]
	
	print idx,lv1_code,lv2_code
	for name in names:
		writer.writerow( [idx,lv1_code,lv2_code,name,'2015-04-01 14:52:32','2015-04-01 14:52:32'] )
		idx +=1

	
	
