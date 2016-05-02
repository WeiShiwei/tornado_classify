#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import fnmatch
import commands

def traverse_directory_tree(root, patterns='*.train', single_level=False, yield_folders=False):
    multilevelCode_filePath_dict = dict()

    patterns = patterns.split(';')
    for path, subdirs, files in os.walk(root):
        if yield_folders:
            files.extend(subdirs)
        files.sort()
        for name in files:
            for pattern in patterns:
                if fnmatch.fnmatch(name,pattern):
                    multilevelCode_filePath_dict[ os.path.basename(path) ] = os.path.join(path, name)
                    break
        if single_level:
            break
    
    # {'2303': '/home/weishiwei/scikit_learn_data/23/2303/23.2303.train'}
    return multilevelCode_filePath_dict



def main():
    source_dir = os.path.expanduser( os.path.join('~','scikit_learn_data'))
    temp_dir = os.path.expanduser( os.path.join('~','classify_learn_data','gldjc-train'))

    multilevelCode_filePath_dict = traverse_directory_tree(source_dir)
    for code,filepath in multilevelCode_filePath_dict.items():
        lv1_code,lv2_code = code[:2],code[2:]

        # lv3_codes = ['01','02','03']
        # for lv3_code in lv3_codes:
        #     pass

        hierarchical_code = lv1_code+'.'+lv2_code

        # dest_dir = os.path.join(temp_dir, hierarchical_code)
        dest_dir = os.path.join(temp_dir, lv1_code, lv2_code)
        if not os.path.exists( dest_dir ):
            os.makedirs( dest_dir )

        src = filepath;dst = dest_dir
        commandLine = 'cp '
        commandLine += src+' '
        commandLine += dst
        print commandLine
        (status, output)=commands.getstatusoutput(commandLine)




if __name__ == '__main__':
	main()