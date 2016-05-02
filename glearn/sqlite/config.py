#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
"""开发环境使用的配置
"""
DB_DEBUG = False
# DB = os.path.join( os.path.abspath(os.path.dirname(__file__)) , '../..' , 'db' , 'glodon_db')
DB = os.path.join( os.path.abspath(os.path.dirname(__file__)) , 'db' , 'auth_db')
HOST = '127.0.0.1'
USER = 'glodon'
PASSWD = 'glodon'
CONNECT_STRING = 'sqlite:///' + DB
