spark-submit \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python \
    --conf spark.executorEnv.PYSPARK_PYTHON=./environment/bin/python \
    --conf spark.task.cpus=4 \
    --archives ./environment.tar.gz#environment \
    --master yarn \
    --queue root.users.root \
    --conf spark.driver.extraClassPath=./deps/assembly/* \
    --conf spark.executor.extraClassPath=./deps/assembly/* \
    --py-files ./deps/python/bigdl-spark_2.4.6-2.1.0-SNAPSHOT-python-api.zip \
    --jars ./deps/assembly/bigdl-dllib-spark_2.4.6-2.1.0-SNAPSHOT-jar-with-dependencies.jar,./deps/assembly/bigdl-orca-spark_2.4.6-2.1.0-SNAPSHOT-jar-with-dependencies.jar \
    --deploy-mode cluster \
    --executor-memory 40g \
    --driver-memory 10g \
    --executor-cores 4 \
    --num-executors 32 \
    xgb.py
