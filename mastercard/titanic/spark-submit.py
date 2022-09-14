export SPARK_HOME=/opt/work/spark-2.4.6-bin-hadoop2.7
export PYTHON_PATH=./environment/bin/python
export PYTHON_PATH_LOCAL=~/anaconda3/envs/orca-zcg/bin/python
${SPARK_HOME}/bin/spark-submit \
    --conf spark.executorEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64 \
    --conf spark.pyspark.python=${PYTHON_PATH} \
    --conf spark.pyspark.driver.python=${PYTHON_PATH_LOCAL} \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=${PYTHON_PATH} \
    --conf spark.executorEnv.PYSPARK_PYTHON=${PYTHON_PATH} \
    --archives ./environment.tar.gz#environment \
    --master yarn \
    --queue root.users.root \
    --py-files /home/kai/zcg/example/bigdl-assembly-spark_2.4.6-2.0.0.zip \
    --conf spark.driver.extraClassPath=/home/kai/zcg/example/jars/* \
    --conf spark.executor.extraClassPath=/home/kai/zcg/example/jars/* \
    --jars /home/kai/anaconda3/envs/orca-zcg/lib/python3.7/site-packages/bigdl/share/dllib/lib/bigdl-dllib-spark_2.4.6-2.0.0-jar-with-dependencies.jar,/home/kai/anaconda3/envs/orca-zcg/lib/python3.7/site-packages/bigdl/share/orca/lib/bigdl-orca-spark_2.4.6-2.0.0-jar-with-dependencies.jar \
    --deploy-mode client \
    --executor-memory 40g \
    --driver-memory 40g \
    --executor-cores 4 \
    --num-executors 2 \
    titanic_large.py

