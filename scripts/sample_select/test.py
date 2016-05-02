# -*- coding: utf-8 -*-
import os,sys
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../'))

from glearn.datasets import load_files



multilevel_code = '01.0101'
categories_select = None
data_train = load_files.fetch_Bunch_datas(multilevel_code,categories=categories_select)
import pdb;pdb.set_trace()
