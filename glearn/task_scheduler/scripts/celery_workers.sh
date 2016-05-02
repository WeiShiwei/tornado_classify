#!/bin/sh

# celery multi命令-A参数指定了任务的名称，
# 比如glearn.task_scheduler.tasks_classify.gldjc模块下的predict的任务就是
# glearn.task_scheduler.tasks_classify.gldjc.predict
# 需要shell在tornado_classify/当前目录中执行脚本

runPath="/home/weishiwei/var/run/celery"
logPath="/home/weishiwei/var/log/celery"

if [ ! -d "$runPath" ]; then 
  mkdir "$runPath" 
fi

if [ ! -d "$logPath" ]; then 
  mkdir "$logPath" 
fi

cmd=$1
classifyPath="/home/weishiwei/800w_classifier/tornado_classify"
start(){
  echo "start celery workers ..."
  
  cd "$classifyPath"

  # In the background（多节点）
  celery multi start gldjc_worker -A glearn.task_scheduler.tasks_classify.gldjc -Q gldjc -l info --concurrency=1 \
  --pidfile=/home/weishiwei/var/run/celery/%n.pid \
  --logfile=/home/weishiwei/var/log/celery/%n%I.log 

  celery multi start gldzb_worker -A glearn.task_scheduler.tasks_classify.gldzb -Q gldzb -l info --concurrency=1 \
  --pidfile=/home/weishiwei/var/run/celery/%n.pid \
  --logfile=/home/weishiwei/var/log/celery/%n%I.log 
  # pid=`ps -ef | grep -v grep | grep -v "lighttpd.sh" | grep lighttpd | sed -n '1P' | awk '{print $2}'`
  # if [ -z $pid ] ; then
  #   /usr/local/lighttpd/src/lighttpd -f /usr/local/lighttpd/doc/lighttpd.conf #lighttpd 启动路径
  # else
  #   echo "lighttpd is running!"
  # fi
} 


stop(){
  echo "killing celery workers ..." 

  cd "$classifyPath"

  # In the background（多节点）
  celery multi stop gldjc_worker -A tasks_classify.gldjc -Q gldjc -l info --concurrency=1 \
  --pidfile=/home/weishiwei/var/run/celery/%n.pid \
  --logfile=/home/weishiwei/var/log/celery/%n%I.log

  celery multi stop gldzb_worker -A tasks_classify.gldzb -Q gldzb -l info --concurrency=1 \
  --pidfile=/home/weishiwei/var/run/celery/%n.pid \
  --logfile=/home/weishiwei/var/log/celery/%n%I.log
  # pid=`ps -ef | grep -v grep | grep -v "lighttpd.sh" | grep lighttpd | sed -n '1P' | awk '{print $2}'`
  # if [ -z $pid ] ; then 
  #   echo "lighttpd is killing" 
  # else       
  #   killall lighttpd
  # fi
}

restart(){
  stop
  start
}

status(){
  # pid=`ps -ef | grep -v grep | grep -v "lighttpd.sh" | grep lighttpd | sed -n '1P' | awk '{print $2}'`
  # if [ -z $pid ] ; then 
  #   echo "lighttpd is not running" 
  # else       
  #   echo "lighttpd is  running" 
  # fi
  echo "celery workers status..."
}

help(){
  echo "Usage: $0 {start|stop|restart|status}"
}
    
case ${cmd} in
  [Ss][Tt][Aa][Rr][Tt])
    start
    ;; 
  [Ss][Tt][Oo][Pp]) 
    stop
    ;;
  [Rr][Ee][Ss][Tt][Aa][Rr][Tt])
    restart
    ;; 
  [Hh][Ee][Ll][Pp])
    help
    ;;
  [Ss][Tt][Aa][Tt][Uu][Ss])
    status
    ;;
  *)
    echo "please read stop or start!"
    ;;
esac
