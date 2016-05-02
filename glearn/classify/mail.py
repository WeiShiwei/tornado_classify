#!/usr/bin/env python
# -*- coding: utf-8 -*-
#导入smtplib和MIMEText
import smtplib,sys 
from email.mime.text import MIMEText 
import pdb


class Mail(object):
    """docstring for Mail"""
    def __init__(self,  mail_host='smtp.163.com',
                        mail_user="weishiwei931@163.com",
                        mail_pass="redemption931",
                        mail_postfix="163.com",

        ):
        super(Mail, self).__init__()
        self.mail_host = mail_host
        self.mail_user = mail_user
        self.mail_pass = mail_pass
        self.mail_postfix = mail_postfix


    def send_mail(self, sub, content, mailto_list='tornado_classify@163.com'): 
        #############
        #要发给谁，这里发给1个人
        # mailto_list=[""] 
        #####################
        #设置服务器，用户名、口令以及邮箱的后缀
        # # mail_user="tornado_classify@163.com"
        # # mail_pass="123qwe!@#"

        # mail_host="smtp.live.com"
        # mail_user="tornado_glearn@hotmail.com"
        # mail_pass="123qwe!@#"
        # mail_postfix="hotmail.com"
        ######################
        '''''
        to_list:发给谁
        sub:主题
        content:内容
        send_mail("aaa@126.com","sub","content")
        '''
        me=self.mail_user+"<"+self.mail_user+"@"+self.mail_postfix+">"
        msg = MIMEText(content,_charset='gbk') 
        msg['Subject'] = sub 
        msg['From'] = me 
        msg['To'] = ";".join(mailto_list) 
        try: 
            # pdb.set_trace()
            s = smtplib.SMTP() 
            s.connect(self.mail_host) 
            s.login(self.mail_user,self.mail_pass) 
            s.sendmail(me, mailto_list, msg.as_string()) 
            s.close() 
            return True
        except Exception, e: 
            print str(e) 
            return False

if __name__ == '__main__': 
    mailto_list = ['tornado_classify@163.com']
    if Mail().send_mail( 'Text classification - Training Model Results','content', mailto_list = mailto_list): 
        print u'发送成功'
    else: 
        print u'发送失败'
