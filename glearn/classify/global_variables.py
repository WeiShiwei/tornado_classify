#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, glob
import hashlib
BLOCKSIZE = 65536

from os import environ
from os.path import dirname
from os.path import join
from os.path import exists
from os.path import expanduser
from os.path import isdir
from os import listdir
from os import makedirs

# AUTHENTICATED_IDENTITIS = ["gcj",'zb']
# GLDJC_IDENTITY = 'gldjc'
# -----------------------------------------------------
ENV = os.environ.get('API_ENV', 'development')
if ENV == 'development':
    # for development
    MAIN_USER = 'weishiwei'
else:
    # for production
    MAIN_USER = 'glearn'
RESOURCES_PRODUCT_PATH = os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../resources/products' )

CORPUS_PATH = '/home/'+MAIN_USER+'/scikit_learn_data'
MODELS_PATH = '/home/'+MAIN_USER+'/scikit_learn_joblib'
stop_words_path = os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../../dependence/assets/stopwords.txt' )
custom_dic_path = os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../../dependence/assets/custom.dic' )

base_clf_path = os.path.join( MODELS_PATH , 'LinearSVCClassifier@01_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_50_51_52_53_54_55_56_57_58_80_98_99.joblib' )
# -----------------------------------------------------
# gldjc_archive = 'ml-gldjc-com'
encoding = 'utf-8'
PROCESSES = 12
# -----------------------------------------------------
def get_testdata_home( data_home=None):
    if data_home is None:
        data_home = environ.get('SCIKIT_LEARN_DATA',
                                join('~', 'classify_learn_testdata'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home

def get_data_home( data_home=None):
    """Return the path of the scikit-learn data dir.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.

    By default the data dir is set to a folder named 'scikit_learn_data'
    in the user home folder.

    Alternatively, it can be set by the 'SCIKIT_LEARN_DATA' environment
    variable or programmatically by giving an explit folder path. The
    '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.
    """
    if data_home is None:
        data_home = environ.get('SCIKIT_LEARN_DATA',
                                join('~', 'classify_learn_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home

def get_model_home( model_home=None):
    """Return the path of the scikit-learn data dir.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.

    By default the data dir is set to a folder named 'scikit_learn_data'
    in the user home folder.

    Alternatively, it can be set by the 'SCIKIT_LEARN_DATA' environment
    variable or programmatically by giving an explit folder path. The
    '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.
    """
    if model_home is None:
        model_home = environ.get('SCIKIT_LEARN_MODEL',
                                join('~', 'classify_learn_model'))
    model_home = expanduser(model_home)
    if not exists(model_home):
        makedirs(model_home)
    return model_home


if __name__ == '__main__':
    print get_data_home()
    print get_model_home('scikit_learn_joblib')
