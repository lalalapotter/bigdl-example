spark-submit \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python \
    --conf spark.executorEnv.PYSPARK_PYTHON=./environment/bin/python \
    --conf spark.task.cpus=4 \
    --archives ./environment.tar.gz#environment \
    --master yarn \
    --queue root.users.root \
    --conf spark.driver.extraClassPath=./deps/assembly/* \
    --conf spark.executor.extraClassPath=./deps/assembly/* \
    --py-files ./deps/python/bigdl-spark_2.4.6-2.2.0-python-api.zip \
    --jars ./deps/jars/bigdl-assembly-spark_2.4.6-2.2.0-jar-with-dependencies.jar \
    --deploy-mode cluster \
    --executor-memory 40g \
    --driver-memory 10g \
    --executor-cores 4 \
    --num-executors 32 \
    xgb_clf.py
