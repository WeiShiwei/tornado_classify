# -*- coding: utf-8 -*-
import os
import sys
import shutil
import glob
import ujson

import traceback
from api.handler.base_handler import BaseHandler

from lib.log_util import create_log
app_logger = create_log('classify')

from glearn.classify.mail import Mail
from glearn.classify import global_variables as gv
from glearn.classify.linear_svc import LinearSVCClassifier
from glearn.classify.classify_learn_platform import ClassifyLearn
from glearn.sqlite.session import authenticate_identity
from glearn.task_scheduler import TaskScheduler

import tornado

class ApplyAsyncHandler(BaseHandler):
    _label = "ApplyAsyncHandler"

    @tornado.web.asynchronous
    @tornado.gen.engine
    def post(self):
        try:
            identity = self.get_argument('identity', default='')
            docs = ujson.loads( self.get_argument('docs',default='') )
            
            res = TaskScheduler.apply_async(identity, docs)
            if res is None:
                json_result = {
                    "task_id": None,
                    "task_name": None,
                    "status": None,
                    "result": None # None
                }
                self._json_response(json_result)
            #\ result = yield tornado.gen.Task(self.application.conn.get, 'celery-task-meta-'+res.id)

            json_result = {
                "task_id": res.id,
                "task_name": res.task_name,
                "status": res.status,
                "result": res.result # None
            }
            self._json_response(json_result)
        except Exception, e:
            self.send_error()
            self._app_logger.error(traceback.format_exc())

class CeleryResultHandler(BaseHandler):
    _label = "CeleryResultHandler"

    def post(self):
        try:
            task_id = self.get_argument('task_id', default='')
            json_result = self.application.rc.get('celery-task-meta-'+task_id)
            self._json_response(json_result)
        except Exception, e:
            self.send_error()
            self._app_logger.error(traceback.format_exc())

class CeleryMonitoringHandler(BaseHandler):
    _label = "CeleryMonitoringHandler"

    def get(self):
        self.redirect('http://localhost:5555/')


class UploadFileHandler(BaseHandler):
    _label = "UploadFileHandler"
    
    def get(self):
        self.render('learn.html')
 
    def post(self):
        try: 
            data_home = gv.get_data_home()#文件的暂存路径  
                      
            identity = self.get_argument('identity', default='')
            email_address = self.get_argument('email', default='')
            load_files_type = self.get_argument('load_files_type', default='')
            model_learning_type = self.get_argument('model_learning_type', default='')

            if not authenticate_identity( identity ):
                self.write('Authentication failed!')
                self.finish() 
                return 

            file_metas=self.request.files['file']    #提取表单中‘name’为‘file’的文件元数据
            for meta in file_metas:
                archive_file_name=meta['filename'] # meta <= <class 'tornado.httputil.HTTPFile'>
                
                identity_home = os.path.join( data_home, identity )
                if os.path.exists( identity_home ):
                    shutil.rmtree( identity_home )
                os.makedirs( identity_home )
                with open( os.path.join( identity_home, archive_file_name ),'wb' ) as up:   # 有些文件需要已二进制的形式存储，实际中可以更改
                    up.write(meta['body'])

                try:
                    # >>>>>>>>
                    fit_sys_stdout, clfs = ClassifyLearn.learn( identity, archive_file_name, 
                                                        load_files_type = load_files_type)
                except Exception, e:
                    self.write( 'ClassifyFit learn error!\n ')
                    self.write( str(e) )
                    self.finish() 
                    return 
            
            # -*- 把模型训练过程中打印的信息以邮件的形式发送给用户 -*-
            email_content = list()
            email_content.append( "Type: " + model_learning_type )
            email_content.append( "Identity: " + identity )
            email_content.append( fit_sys_stdout )
            mailto_list = [addr.strip() for addr in email_address.split(';')]
            if Mail().send_mail( 'Text classification - Training Model Results','\n'.join(email_content), mailto_list = mailto_list): 
                print u'发送成功'
            else: 
                print u'发送失败'

            json_result = {
                "identity":identity,
                "type":model_learning_type,
                "status":'SUCCESS'
            }
            self._json_response( json_result )
        except:
            self.send_error()
            self._app_logger.error(traceback.format_exc())      
