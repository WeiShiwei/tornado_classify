#!/usr/bin/python
# -*- coding: utf-8 -*-

import os,sys
import csv
import shutil
from os import environ
from os.path import dirname
from os.path import join
from os.path import exists
from os.path import expanduser
from os.path import isdir
from os import listdir
from os import makedirs
import fnmatch
import collections
from time import time
import glob

import numpy as np
from sklearn.utils import check_random_state
import sklearn

class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def _traverse_directory_tree(root, pattern='*.train'):
    trainfiles = list()
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch.fnmatch(name,pattern):
                trainfiles.append( os.path.join(path,name) )
    return trainfiles

def traverse_directory_tree(root, patterns='*.train', single_level=True, yield_folders=False, multilevel_codes=None):
    multilevelCode_filePath_dict = dict()

    patterns = patterns.split(';')
    for path, subdirs, files in os.walk(root):
        for subdir in subdirs:
            subdir_absolute = os.path.join(path,subdir)
            trainfiles = _traverse_directory_tree(subdir_absolute)
            if multilevel_codes:
                hierarchical_code = '.'.join( multilevel_codes )+'.'+subdir
                hierarchical_code = hierarchical_code.strip('.') # in case of multilevel_codes == ['']
            else:
                hierarchical_code = '.'.join( multilevel_codes )
            multilevelCode_filePath_dict[ hierarchical_code ] = trainfiles

        if single_level:
            break
    return multilevelCode_filePath_dict

def load_files(data_home, multilevel_codes=None, description=None, categories=None,
               load_content=True, shuffle=True, encoding=None,
               decode_error='strict', random_state=0,
               feature_engineering=True, load_files_type='document',
               hierarchical=None):
    """Load text files with categories as subfolder names.

    Individual samples are assumed to be files stored a two levels folder
    structure such as the following:

        container_folder/
            category_1_folder/
                file_1.txt
                file_2.txt
                ...
                file_42.txt
            category_2_folder/
                file_43.txt
                file_44.txt
                ...
    ...
    Parameters
    -------
    data_home = /home/weishiwei/classify_learn_data/gldjc/gldjc-train-01-05/gldjc-train
    multilevel_codes = ['01','01']

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are: either
        data, the raw text data to learn, or 'filenames', the files
        holding it, 'target', the classification labels (integer index),
        'target_names', the meaning of the labels, and 'DESCR', the full
        description of the dataset.
    """
    container_path = data_home
    if multilevel_codes:
        for code in multilevel_codes:
            container_path = os.path.join(container_path, code)
    print "container_path:", container_path
    
    # load_files_type == 'document'
    if load_files_type == 'document':
        data_train = sklearn.datasets.load_files(container_path, description=description, categories=categories,
                                   load_content=load_content, shuffle=shuffle, encoding=encoding,
                                   decode_error=decode_error, random_state=random_state,
                                   # feature_engineering = feature_engineering,
                                   )
        return data_train

    # load_files_type == 'line'
    multilevelCode_filePath_dict = traverse_directory_tree( container_path ,multilevel_codes=multilevel_codes)
    container_path = multilevelCode_filePath_dict

    target = []
    target_names = []
    filenames = []
    # -----------------------------------------
    data_line_mode = list()
    # -----------------------------------------
    print container_path
    # folders = [f for f in sorted(listdir(container_path))
    #            if isdir(join(container_path, f))]
    folders = sorted(container_path.keys())
    if categories is not None:
        folders = [f for f in folders if f in categories]

    for label, folder in enumerate(folders):# label是folder(category)的索引
        target_names.append(folder)
        # folder_path = join(container_path, folder)
        # documents = [join(folder_path, d)
        #              for d in sorted(listdir(folder_path))]
        documents = container_path.get(folder)
        # -------------------------------------------------------------
        # target.extend(len(documents) * [label])
        
        for doc in documents:
            lines = map(lambda x:x.strip(), open(doc,'rb').readlines())
            data_line_mode.extend(lines)
            target.extend(len(lines) * [label])
        # -------------------------------------------------------------
        filenames.extend(documents)

    # convert to array for fancy indexing
    filenames = np.array(filenames)
    target = np.array(target)

    # -------------------------------------------------------------
    # 下面注释掉的一段代码会把target的维度信息弄丢，所以暂时注释掉
    # import pdb;pdb.set_trace()
    # if shuffle:
    #     random_state = check_random_state(random_state)
    #     indices = np.arange(filenames.shape[0])
    #     random_state.shuffle(indices)
    #     filenames = filenames[indices]
    #     target = target[indices]
    # -------------------------------------------------------------

    if load_content:
    	# -------------------------------------------------------------
        # data = [open(filename, 'rb').read() for filename in filenames]
        # if encoding is not None:
        #     data = [d.decode(encoding, decode_error) for d in data]

        data = data_line_mode
        if encoding is not None:
            data = [d.decode(encoding, decode_error) for d in data]
        
        # if feature_engineering:
        #     data = [Feature_Engineering.normalize(d) for d in data] # <================ shutdown 特征工程
        # -------------------------------------------------------------
        
        return Bunch(data=data,
                     filenames=filenames,
                     target_names=target_names,
                     target=target,
                     DESCR=description)

    return Bunch(filenames=filenames,
                 target_names=target_names,
                 target=target,
                 DESCR=description)


def main():
    multilevel_code = '01'
    categories_select = None

    data_home = '/home/weishiwei/classify_learn_data/gldjc-train'
    multilevel_codes = ['01']

    data_train = load_files(data_home,
                            multilevel_codes=multilevel_codes,
                            encoding='utf-8',
                            feature_engineering = True,
                            load_files_type='line')
    
    import pdb;pdb.set_trace()
    # info of data_train
    # print('data loaded')
    # categories = data_train.target_names    # for case categories == None
    # def size_mb(docs):
    #     return sum(len(s.encode('utf-8')) for s in docs) / 1e6
    # data_train_size_mb = size_mb(data_train.data)       
    # print("%d documents - %0.3fMB (training set)" % (
    #     len(data_train.data), data_train_size_mb))
    # print("%d categories" % len(categories))
    # print()

    # import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()



