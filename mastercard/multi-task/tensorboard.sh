export JAVA_HOME=/usr/java/jdk1.8.0_181-amd64
export LD_LIBRARY_PATH=/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64:${JAVA_HOME}/jre/lib/amd64/server
export CLASSPATH=$(hadoop classpath --glob)

tensorboard --logdir hdfs://default/user/kai/zcg/logs --bind_all

