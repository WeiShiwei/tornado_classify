#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '..'))

import tornado
import tornado.gen
import tornado.httpserver
from tornado import ioloop, web
from tornado.options import define, parse_command_line, options
from lib.log_util import create_log

from api.config import config
from api.handler import *

import redis

define('port', default=9711, help="run on the given port", type=int)

class Application(tornado.web.Application):
    def __init__(self):
        self.rc = redis.StrictRedis(host='localhost', port=6379, db=0)
        
        handlers = [
            # (r'/classify', ClassifyIndexHandler),
            (r'/classify/predict', ClassifyPredictHandler),
            # (r'/classify/test', ClassifyTestPredictHandler),
            (r'/classify/test/upload', ClassifyTestUploadFileHandler),


            # (r'/file', UploadFileHandler),# 禁用上传
            (r'/tasks/apply-async', ApplyAsyncHandler),
            (r'/tasks/result', CeleryResultHandler),

            (r'/celery/monitoring', CeleryMonitoringHandler)
        ]
            
        settings = {
            'template_path': os.path.join(os.path.dirname(__file__), "templates"),
            'static_path': os.path.join(os.path.dirname(__file__), "static"),
            'debug': True
        }
        
        tornado.web.Application.__init__(self, handlers, **settings)

if __name__ == '__main__':
    app_logger = create_log('api')

    tornado.options.parse_command_line()

    app = Application()
    server = tornado.httpserver.HTTPServer(app)
    server.listen(options.port)
    app_logger.info('tornado_crf API listen on %d' % options.port)

    tornado.ioloop.IOLoop.instance().start()

