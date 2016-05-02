from sqlalchemy import Column, Integer, Unicode, ForeignKey, Boolean
from model.base import Base
from sqlalchemy.orm import sessionmaker, relationship, mapper



class BaseMaterialTypeProductNames(Base):
    __tablename__ = 'base_material_type_product_names'
    
    id = Column(Integer, primary_key=True)
    first_type_code = Column(Unicode(4), nullable=True)
    second_type_code = Column(Unicode(4), nullable=True)
    description = Column(Unicode(128), nullable=True)
    