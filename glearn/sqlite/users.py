from sqlalchemy import Column, Integer, Unicode, ForeignKey, Boolean
from base import Base
from sqlalchemy.orm import sessionmaker, relationship, mapper


class Users(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(Unicode(128), nullable=True)
    email_addr = Column(Unicode(128), nullable=True)
        