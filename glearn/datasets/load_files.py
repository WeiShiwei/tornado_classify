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

import numpy as np
from sklearn.utils import check_random_state

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../../'))
# from glearn.feature_engineering import Feature_Engineering
from glearn.classify import global_variables as gv

# ---------------------------------mtime:2014/12/09-------------------------------------------------
# HOME_DIR = gv.CORPUS_PATH
# HOME_DIR = os.path.join( gv.get_data_home(), 'gldjc', gv.gldjc_archive, gv.gldjc_archive+'-train')
# print 'HOME_DIR:',HOME_DIR
# ----------------------------------------------------------------------------------

class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def traverse_directory_tree(root,patterns='*.train;*.joblib',
                            single_level=False,
                            yield_folders=False):
    multilevelCode_filePath_dict = dict()

    patterns = patterns.split(';')
    for path, subdirs, files in os.walk(root):
        if yield_folders:
            files.extend(subdirs)
        files.sort()
        for name in files:
            for pattern in patterns:
                if fnmatch.fnmatch(name,pattern):
                    # print os.path.join(path, name)
                    # multilevelCode_filePath_dict[name[0:(name.index('.train'))]] = os.path.join(path, name)
                    multilevelCode_filePath_dict[name[0:(name.rindex('.'))]] = os.path.join(path, name)
                    break
        if single_level:
            break
    # for (key,value) in multilevelCode_filePath_dict.items():
    #     print key+":"+value
    return multilevelCode_filePath_dict

# ---------------------------------mtime:2014/12/09-------------------------------------------------
def get_curLv2Code_filePaths(multilevelCode_filePath_dict, code):
    lv2Code_filePath_dict = collections.defaultdict(list)

    # import pdb;pdb.set_trace()
    for key in multilevelCode_filePath_dict.keys():
        if key.startswith(code):
            if code.find('.') is not -1: #三级分类？
                curLv2Code = code
                lv2Code_filePath_dict[curLv2Code].append(multilevelCode_filePath_dict.get(key))
                continue
            remainder = key[len(code):]
            curLv2Code = remainder.strip('.').split('.')[0]
            lv2Code_filePath_dict[curLv2Code].append(multilevelCode_filePath_dict.get(key))
    
    # for (key,value) in lv2Code_filePath_dict.items():
    #     print key+":"+';'.join(value)
    return lv2Code_filePath_dict

# def __get_curLv2Code_filePaths(multilevelCode_filePath_dict, code):
#     lv2Code_filePath_dict = collections.defaultdict(list)
#     for key in multilevelCode_filePath_dict.keys():
#         lv2Code_filePath_dict[key].append(multilevelCode_filePath_dict.get(key))
#     return lv2Code_filePath_dict
# ----------------------------------------------------------------------------------

def fetch_Bunch_datas(multilevel_code, description="basic_datas", 
                    categories=None, is_for_train=True,
                    feature_engineering = True):
    """"""
    t0 = time()
    multilevelCode_filePath_dict = traverse_directory_tree( gv.CORPUS_PATH )# <HOME_DIR>multilevel_code='25'或者为空(空代表主目录)
    lv2Code_filePath_dict = get_curLv2Code_filePaths(multilevelCode_filePath_dict, multilevel_code)
    Bunch_data = load_files(lv2Code_filePath_dict,
                        description=description, 
                        encoding='utf-8',
                        categories=categories,
                        feature_engineering = True)
    load_files_time = time() - t0
    print("load_files_ time: %0.3fs" % load_files_time)
    
    if is_for_train:
        print('data loaded')
        categories = Bunch_data.target_names    # for case categories == None
        def size_mb(docs):
            return sum(len(s.encode('utf-8')) for s in docs) / 1e6
        data_train_size_mb = size_mb(Bunch_data.data)       
        print("%d documents - %0.3fMB (training set)" % (
            len(Bunch_data.data), data_train_size_mb))
        print("%d categories" % len(categories))
        print()
    else:
        # borrow a path
        pass

    return Bunch_data

def fetch_hierarchy_code_list(multilevel_code = '', categories = None, feature_engineering = False):
    # multilevel_code = ''
    # categories = None
    data_train = fetch_Bunch_datas(multilevel_code = multilevel_code, categories = categories)
    return data_train.target_names
# def fetch_hierarchy_code_list_secondary(multilevel_code = '',):

def fetch_fine_grained_level_datas(multilevel_code, description="basic_datas", categories=None, is_for_train=True):
    """"""
    t0 = time()
    # import pdb;pdb.set_trace()
    multilevelCode_filePath_dict = traverse_directory_tree( gv.CORPUS_PATH )
    # multilevel_code='25'或者为空(空代表主目录)
    # import pdb;pdb.set_trace()
    # lv2Code_filePath_dict = get_curLv2Code_filePaths(multilevelCode_filePath_dict, multilevel_code)
    # Bunch_data = load_files(lv2Code_filePath_dict,description=description, encoding='utf-8',categories=categories)
    lv2Code_filePath_dict = dict()
    for (key,value) in multilevelCode_filePath_dict.items():
        # temp = list()
        lv2Code_filePath_dict[key] = list()
        lv2Code_filePath_dict[key].append(value)
    # import pdb;pdb.set_trace()
    Bunch_data = load_files(lv2Code_filePath_dict,description=description, encoding='utf-8',categories=categories)
    load_files_time = time() - t0
    print("load_files_ time: %0.3fs" % load_files_time)
    
    if is_for_train:
        print('data loaded')
        categories = Bunch_data.target_names    # for case categories == None
        # self.__set_categories(categories) 
        def size_mb(docs):
            return sum(len(s.encode('utf-8')) for s in docs) / 1e6
        data_train_size_mb = size_mb(Bunch_data.data)       
        print("%d documents - %0.3fMB (training set)" % (
            len(Bunch_data.data), data_train_size_mb))
        print("%d categories" % len(categories))
        print()
    else:
        # borrow a path
        pass

    return Bunch_data


# def fetch_data_train()


def load_files(container_path, description=None, categories=None,
               load_content=True, shuffle=True, encoding=None,
               decode_error='strict', random_state=0,
               feature_engineering = True):
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

    The folder names are used as supervised signal label names. The
    individual file names are not important.

    This function does not try to extract features into a numpy array or
    scipy sparse matrix. In addition, if load_content is false it
    does not try to load the files in memory.

    To use text files in a scikit-learn classification or clustering
    algorithm, you will need to use the `sklearn.feature_extraction.text`
    module to build a feature extraction transformer that suits your
    problem.

    If you set load_content=True, you should also specify the encoding of
    the text using the 'encoding' parameter. For many modern text files,
    'utf-8' will be the correct encoding. If you leave encoding equal to None,
    then the content will be made of bytes instead of Unicode, and you will
    not be able to use most functions in `sklearn.feature_extraction.text`.

    Similar feature extractors should be built for other kind of unstructured
    data input such as images, audio, video, ...

    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category

    description: string or unicode, optional (default=None)
        A paragraph describing the characteristic of the dataset: its source,
        reference, etc.

    categories : A collection of strings or None, optional (default=None)
        If None (default), load all the categories.
        If not None, list of category names to load (other categories ignored).

    load_content : boolean, optional (default=True)
        Whether to load or not the content of the different files. If
        true a 'data' attribute containing the text information is present
        in the data structure returned. If not, a filenames attribute
        gives the path to the files.

    encoding : string or None (default is None)
        If None, do not try to decode the content of the files (e.g. for
        images or other non-text content).
        If not None, encoding to use to decode text files to Unicode if
        load_content is True.

    decode_error: {'strict', 'ignore', 'replace'}, optional
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. Passed as keyword
        argument 'errors' to bytes.decode.

    shuffle : bool, optional (default=True)
        Whether or not to shuffle the data: might be important for models that
        make the assumption that the samples are independent and identically
        distributed (i.i.d.), such as stochastic gradient descent.

    random_state : int, RandomState instance or None, optional (default=0)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are: either
        data, the raw text data to learn, or 'filenames', the files
        holding it, 'target', the classification labels (integer index),
        'target_names', the meaning of the labels, and 'DESCR', the full
        description of the dataset.
    """
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
        # print label,folder
        target_names.append(folder)
        # folder_path = join(container_path, folder)
        # documents = [join(folder_path, d)
        #              for d in sorted(listdir(folder_path))]
        documents = container_path.get(folder)
        # -------------------------------------------------------------
        # target.extend(len(documents) * [label])
        
        for doc in documents:
            lines = open(doc,'rb').readlines()
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





