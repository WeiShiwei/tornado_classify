#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import config
from users import Users


# if ENV == 'development':
engine = create_engine(
    config.CONNECT_STRING,
    echo=config.DB_DEBUG
)
# else:
#     engine = create_engine(
#         config.CONNECT_STRING,
#         echo=config.DB_DEBUG,
#         pool_recycle=3600,
#         pool_size=15
#     )

Session = sessionmaker(bind=engine)


@contextlib.contextmanager
def get_session():
    """
    session 的 contextmanager， 用在with语句
    """
    session = Session()
    try:
        yield session
    except Exception as e:
        session.rollback()
        print 'CANT GET SESSION, ERROR: '
        print e
        raise
    finally:
        session.close()

def authenticate_identity(identity):
	with get_session() as session:
		# import pdb;pdb.set_trace()
		user = session.query( Users ).filter(Users.name == identity).first()
		if not user:
			return False
		else:
			return True



def main():
	# import pdb;pdb.set_trace()
	identity = 'gldjc'
	print authenticate_identity(identity)
	with get_session() as session:
		# print [ (u.name,u.email_addr) for u in session.query( Users ).all()]
		user = session.query( Users ).filter(Users.name == identity).first()
		if not user:
			return None
		else:
			return user.name,user.email_addr

if __name__ == '__main__':
	print main()