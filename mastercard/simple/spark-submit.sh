export PYTHON_PATH=environment/bin/python
spark-submit \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=${PYTHON_PATH} \
    --conf spark.executorEnv.PYSPARK_PYTHON=${PYTHON_PATH} \
    --archives ./environment.tar.gz#environment \
    --master yarn \
    --queue root.users.root \
    --deploy-mode cluster \
    --executor-memory 5g \
    --driver-memory 1g \
    --executor-cores 3 \
    --num-executors 1 \
    --conf spark.yarn.appMasterEnv.HADOOP_USER_NAME=kai \
    --conf spark.executor.memoryOverhead=1024 \
    --conf spark.python.worker.reuse=false \
    --py-files ./deps/python/bigdl-spark_2.4.6-2.2.0-python-api.zip \
    --jars ./deps/jars/bigdl-assembly-spark_2.4.6-2.2.0-jar-with-dependencies.jar \
    train.py