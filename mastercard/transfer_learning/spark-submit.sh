spark-submit \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python \
    --conf spark.executorEnv.PYSPARK_PYTHON=./environment/bin/python \
    --archives ./environment.tar.gz#environment \
    --master yarn \
    --queue root.users.root \
    --deploy-mode cluster \
    --executor-memory 5g \
    --driver-memory 1g \
    --executor-cores 3\
    --num-executors 1 \
    --conf spark.yarn.appMasterEnv.HADOOP_USER_NAME=kai \
    --conf spark.executor.memoryOverhead=1024 \
    --jars /home/kai/anaconda3/envs/orca-nightly-zcg/lib/python3.7/site-packages/bigdl/share/orca/lib/bigdl-orca-spark_2.4.6-2.2.0-jar-with-dependencies.jar,/home/kai/anaconda3/envs/orca-nightly-zcg/lib/python3.7/site-packages/bigdl/share/dllib/lib/bigdl-dllib-spark_2.4.6-2.2.0-jar-with-dependencies.jar,/home/kai/anaconda3/envs/orca-nightly-zcg/lib/python3.7/site-packages/bigdl/share/core/lib/all-2.2.0.jar \
    transfer_learning.py