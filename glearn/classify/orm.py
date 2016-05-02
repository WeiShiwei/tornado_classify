#!/usr/bin/python
# #_*_ coding: utf-8 _*_
from datetime import datetime

from sqlalchemy import update
from sqlalchemy.orm import aliased

from sqlalchemy import (create_engine, MetaData, Table, Column, Integer, Boolean,
    String, DateTime, Float, ForeignKey, and_)
from sqlalchemy.orm import mapper, relationship, sessionmaker, backref,\
                           joinedload_all
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.sql import func , distinct

# import global_variables as gv
import os
#import yaml

import sys
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../..'))

# ---------------------------------------------
from glearn.classify.config import config, ENV
if ENV == 'development':
    engine = create_engine(
        config.CONNECT_STRING, # PostgreSQL数据库ml_2013（本地）
        echo=config.DB_DEBUG
    )
else:
    engine = create_engine(
        config.CONNECT_STRING, # 结构化新平台的数据库db_structure_glodon_com
        echo=config.DB_DEBUG,
        pool_recycle=3600,
        pool_size=15
    )
Base = declarative_base(engine)
session = sessionmaker(bind=engine)()

# ENV = os.environ.get('API_ENV', 'development')
# if ENV == 'development':
#   # for development
#   db = create_engine('postgresql+psycopg2://postgres:postgres@localhost/ml_2013' , echo=False)
# else:
#   # for production
#   db = create_engine('postgresql+psycopg2://gcj:Gl0147D0n258@192.168.10.14/ml_2013' , echo=False)

# Base = declarative_base(db)
# session = sessionmaker(bind=db)()
# ---------------------------------------------


#  parent_id         :integer
#  code              :string(2)
class BaseMaterialType( Base ):
  __tablename__ = 'base_material_types'
  id = Column(Integer, primary_key=True)
  parent_id = Column( Integer )
  code = Column(String)
  description = Column(String)

  @classmethod
  def all_lv1_codes(cls):
    return [ obj[0] for obj in session.query(distinct(BaseMaterialType.code)).filter( BaseMaterialType.parent_id == None ).all()]
  
  @classmethod
  def all_lv2_codes(cls):
    parent_bmt = aliased(BaseMaterialType)
    bes = session.query(BaseMaterialType, parent_bmt).join((parent_bmt, parent_bmt.id == BaseMaterialType.parent_id)).all()

    lv2_codes = list()
    for be in bes:
      first_type_code = be[1].code
      second_type_code = be[0].code
      if first_type_code == '00':
        continue
      lv2_codes.append( be[1].code+ be[0].code)
    return lv2_codes

  @classmethod
  def all_lv2_codes_of(cls , lv1_code ):
    lv1_id = session.query( distinct( BaseMaterialType.id ) ).filter( BaseMaterialType.code == lv1_code ).filter( BaseMaterialType.parent_id == None )[0]
    return [ obj[0] for obj in session.query(distinct(BaseMaterialType.code)).filter( BaseMaterialType.parent_id == lv1_id ).all() ]
    
class DataModel( Base ):
  __tablename__ = 'data_models'
  id = Column(Integer, primary_key=True)
  status = Column( String(50) , nullable=False)
  model_argument_id = Column( Integer , ForeignKey('model_arguments.id') )
  
  model_argument = relationship('ModelArgument')
  
  train_sample_id = Column( Integer , ForeignKey('train_samples.id') )
  train_sample = relationship('TrainSample')

  def model_save_path(self):
    model_path = os.path.join( gv.model_path , type(self).__name__ , str(self.id) )
    if not os.path.exists(model_path):
      os.makedirs(model_path)
    return model_path

  def trainer_save_path(self):
    return os.path.join( gv.trainer_model_path , type(self).__name__ , str(self.id) )
  
  def threshold_save_path( self ):
    return os.path.join( self.model_save_path , 'thresholds.csv' )


class ModelArgument( Base ):
  __tablename__ = 'model_arguments'
  id = Column(Integer, primary_key=True)
  svm_type = Column( String )
  lf = Column( String )
  gl = Column( String )
  svm_c = Column( String )
  svm_gamma = Column( String )
  auto_select = Column( Boolean )
  train_indexes = Column( String )
  feature_ratio = Column( Float )
  
  data_model = relationship(DataModel)
#    csv_line = [ basic_data.first_type_code or basic_data.second_type_code
#                 basic_data.id ,
#                 basic_data.name.encode('utf-8'),
#                 basic_data.unit.encode('utf-8'),
#                 basic_data.brand.encode('utf-8'),
#                 basic_data.spec.encode('utf-8')
#               ]
  def train_indexes_ary(self):
    indexes = []
    for i in str(self.train_indexes).split(','):
      s = i.strip()
      if s == '1':
        indexes.append( 2 )
      elif s == '2':
        indexes.append( 3 )
      elif s == '3':
        indexes.append( 4 )
      elif s == '4':
        indexes.append( 5 )
    return indexes

  def desc(self):
    return '_'.join( [self.gl , self.lf , self.svm_type] ).strip()
  
class TrainSample( Base ):
  __tablename__ = 'train_samples'
  id = Column(Integer, primary_key=True)
  name = Column( String )
  type_level = Column( Integer )
  data_model = relationship(DataModel)
  
  def basic_data_relations(self):
    return session.query(BasicDataRelation).filter( BasicDataRelation.sample_type == 'TrainSample').filter( BasicDataRelation.sample_id == self.id ).all()
    
  def save_path( self ):
    return os.path.join( gv.samples_path , str(type(self).__name__) , str(self.id) , 'samples.csv' )

# Table name: basic_datas
#  id               :integer          not null, primary key
#  first_type_code  :string(255)
#  second_type_code :string(255)
#  name             :string(255)
#  unit             :string(255)
#  brand            :string(255)
#  spec             :string(255)
#  attrs            :string(255)
#  created_at       :datetime         not null
#  updated_at       :datetime         not null
class BasicData( Base ):
  __tablename__ = 'basic_datas'
  id = Column(Integer, primary_key=True)
  first_type_code = Column( String )
  second_type_code = Column( String )
  name = Column( String )
  unit = Column( String )
  brand = Column( String )
  spec = Column( String )
  attrs = Column( String )
  created_at = Column( DateTime )
  updated_at = Column( DateTime )
  

# Table name: basic_data_relations
#
#  id            :integer          not null, primary key
#  basic_data_id :integer
#  sample_id     :integer
#  sample_type   :string(255)
#  created_at    :datetime         not null
#  updated_at    :datetime         not null

class BasicDataRelation( Base ):
  __tablename__ = 'basic_data_relations'
  id = Column(Integer, primary_key=True)
  sample_id = Column( Integer)
  basic_data_id = Column( Integer , ForeignKey('basic_datas.id'))
  sample_type = Column( String )
  
  basic_data = relationship( 'BasicData' )
  
  def belongs_to_sample(self):
    return session.query(eval(self.sample_type)).filter_by( id=self.sample_id ).first()

class ValidateResult( Base ):
  __tablename__ = 'validate_results'
  id = Column(Integer, primary_key=True)
  status = Column( String )
  model_id = Column( Integer )
  model_type = Column( String )
  indexes = Column( String )
  validate_sample_id = Column( Integer , ForeignKey('validate_samples.id'))
  validate_sample = relationship( 'ValidateSample' )
  #belongs_to data_model or key_word_model
  def belongs_to_model(self):
    return session.query(eval(self.model_type)).filter_by( id=self.model_id ).first()

  def indexes_ary(self):
    xs = []
    for i in str(self.indexes).split(','):
      s = i.strip()
      if s == '1':
        xs.append( 2 )
      elif s == '2':
        xs.append( 3 )
      elif s == '3':
        xs.append( 4 )
      elif s == '4':
        xs.append( 5 )
    return xs
    
  def save_path(self , model=None):
    if model is None:
      model = self.belongs_to_model()
    return os.path.join( model.model_save_path() , str(self.id) + "_result.csv")
  
class ValidateSample( Base ):
  __tablename__ = 'validate_samples'
  id = Column(Integer, primary_key=True)
  name = Column( String )
  type_level = Column( Integer )
  validate_result = relationship( 'ValidateResult' )

  def basic_data_relations(self):
    return session.query(BasicDataRelation).filter( BasicDataRelation.sample_type == 'ValidateSample').filter( BasicDataRelation.sample_id == self.id ).all()

  def save_path( self ):
    return os.path.join( gv.samples_path , str(type(self).__name__) , str(self.id) , 'samples.csv' )

#    t.string   "status",             :default => "new"
#    t.integer  "model_id"
#    t.string   "model_type"
#    t.integer  "validate_sample_id"
#    t.integer  "file_id"
class AnalysisResult(Base):
  __tablename__ = 'analysis_results'
  id = Column(Integer, primary_key=True)
  status = Column(String)
  step = Column(Integer)
  status = Column(String)
  model_id = Column(Integer)
  model_type = Column(String)
  validate_sample_id = Column( Integer , ForeignKey('validate_samples.id'))
  validate_sample = relationship( 'ValidateSample' )
  
  validate_result_id = Column( Integer , ForeignKey('validate_results.id'))
  validate_result = relationship( 'ValidateResult' )
  
  def belongs_to_model(self):
    return session.query(eval(self.model_type)).filter_by( id=self.model_id ).first()
  
  def save_path(self , model=None):
    if model is None:
      model = self.belongs_to_model()
    return os.path.join( model.model_save_path() , str(self.id) + "_analysis.csv" )

class KeyWord(Base):
  __tablename__ = 'key_words'
  id = Column(Integer, primary_key=True)
  name = Column( String )

#    t.string   "first_type_code"
#    t.string   "second_type_code"
#    t.string   "name"
#    t.string   "type"
#    t.float    "weight"
class KeyWordReference( Base ):
  __tablename__ = 'key_word_references'
  id = Column(Integer, primary_key=True)
  first_type_code = Column( String )
  second_type_code = Column( String )
  name = Column( String )
  weight = Column( Float )
  updated_at = Column( DateTime ) ###
  created_at = Column( DateTime ) ###

  @classmethod
  def fetch_earliest_created_time(self):
    latest_create_time = session.query(func.min(self.created_at)).first()
    return latest_create_time[0]
  @classmethod
  def fetch_latest_updated_time(self):
    """ created_at =< updated_at"""
    latest_update_time = session.query(func.max(self.updated_at)).select_from(self).scalar()
    latest_count =  session.query(func.count('*')).select_from(self).scalar()
    return latest_update_time,latest_count

  @classmethod
  def fetch_target_names_lv2(self):
    # import pdb;pdb.set_trace()
    target_names = list()
    records = session.query( distinct(self.second_type_code) ).order_by(KeyWordReference.second_type_code)
    for r in records:
      if r[0]:
        target_names.append(r[0])
      else:
        continue
    return target_names
    print target_names
  @classmethod
  def fetch_target_names_lv1(self):
    # import pdb;pdb.set_trace()
    target_names = list()
    records = session.query( distinct(self.first_type_code) ).order_by(KeyWordReference.first_type_code)
    for r in records:
      if r[0]:
        target_names.append(r[0])
      else:
        continue
    return target_names
    print target_names



  def total_weight( self ):
    kwrs = session.query( KeyWordReference ).filter( KeyWordReference.name== self.name ).filter( KeyWordReference.name != '').all()
    return sum( [ kwr.weight for kwr in kwrs if kwr.weight is not None ] ) + 1

#  create_table "key_word_relations", :force => true do |t|
#    t.integer  "key_word_reference_id"
#    t.integer  "key_word_sample_id"
#    t.datetime "created_at",            :null => false
#    t.datetime "updated_at",            :null => false
#  end
class KeyWordRelation( Base ):
  __tablename__ = 'key_word_relations'
  id = Column(Integer, primary_key=True)
  key_word_reference_id = Column( Integer , ForeignKey('key_word_references.id') )
  key_word_reference = relationship( 'KeyWordReference' )
  
  key_word_sample_id = Column( Integer , ForeignKey('key_word_samples.id') )
  key_word_sample = relationship( 'KeyWordSample' )

#  create_table "key_word_samples", :force => true do |t|
#    t.string   "name"
#    t.integer  "type_level"
#    t.integer  "file_id"
#    t.integer  "user_id"
#    t.datetime "created_at",              :null => false
#    t.datetime "updated_at",              :null => false
#    t.string   "base_material_type_code"
#  end
class KeyWordSample( Base ):
  __tablename__ = 'key_word_samples'
  id = Column(Integer, primary_key=True)
  type_level = Column( Integer )
  key_word_relations = relationship('KeyWordRelation')

  def save_path( self ):
    return os.path.join( gv.samples_path , str(type(self).__name__) , str(self.id) , 'samples.csv' )


#  create_table "key_word_models", :force => true do |t|
#    t.string   "status",               :default => "new"
#    t.integer  "config_file_id"
#    t.integer  "key_file_id"
#    t.integer  "model_file_id"
#    t.integer  "modle_argument_id"
#    t.integer  "key_word_sample_id"
#    t.datetime "created_at",                              :null => false
#    t.datetime "updated_at",                              :null => false
#  end
class KeyWordModel(Base):
  __tablename__ = 'key_word_models'
  id = Column(Integer, primary_key=True)
  status = Column( String )
  model_argument_id = Column( Integer , ForeignKey('model_arguments.id') )
  model_argument = relationship( 'ModelArgument' )
  
  key_word_sample_id = Column( Integer , ForeignKey('key_word_samples.id') )
  key_word_sample = relationship( 'KeyWordSample' )

  def model_save_path(self):
    model_path = os.path.join( gv.model_path , type(self).__name__ , str(self.id) )
    if not os.path.exists(model_path):
      os.makedirs(model_path)
    return model_path

  def trainer_save_path(self):
    return os.path.join( gv.trainer_model_path , type(self).__name__ , str(self.id) )
  
  def threshold_save_path( self ):
    return os.path.join( self.model_save_path , 'thresholds.csv' )

#  base_material_type_code :string(255)
#  model_id                :integer
#  model_type              :string(255)
#  min_threshold           :float
#  max_threshold           :float
#  created_at              :datetime         not null
#  updated_at              :datetime         not null
class ModelCredibility( Base ):
  __tablename__ = 'model_credibilities'
  id = Column(Integer, primary_key=True)
  base_material_type_code = Column( String )
  model_id = Column( Integer )
  model_type = Column( String )
  min_threshold = Column( Float )
  max_threshold = Column( Float )
  thresholds_dict = Column( String )
  
  #belongs_to data_model or key_word_model
  def belongs_to_model(self):
    return session.query(eval(self.model_type)).filter_by( id=self.model_id ).first()
  
  def read_thresholds_dict(self):
    return yaml.load(self.thresholds_dict)
  
  @classmethod
  def lv1(cls):  
    # 返回所有一级类别的‘记录集合‘ 
    return session.query(cls).filter( func.length(cls.base_material_type_code) == 2 ).all()
    
  @classmethod
  def lv2(cls , lv1_code):
    try:
      lv1_code_str = str(int(lv1_code)).rjust(2,'0')
    except:
      return []
    ###若lv1_code='9',lv1_code_str='09'，
    ###删选的条件是前缀为lv1_code_str的二级类
    return session.query(cls).filter( func.length(cls.base_material_type_code) == 4 ).\
      filter( cls.base_material_type_code.like( lv1_code_str + '%' ) ).all()

#data_model = session.query(DataModel).sum(  )
#print data_model.train_sample.basic_data_relations()
#print data_model.model_argument.lf

#train_sample = session.query(TrainSample).first()
#for bdr in train_sample.basic_data_relations:
#  print bdr.basic_data.name
#data_model = train_sample.data_model[0]
#data_model.status = 'running'
#session.add(data_model)
#session.commit()

