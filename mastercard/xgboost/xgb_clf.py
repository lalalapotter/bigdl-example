from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.ml.feature import VectorAssembler

from bigdl.dllib.nnframes.tree_model import *
from bigdl.dllib.utils.log4Error import *
from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext


sc = init_orca_context("spark-submit")

data_filepath = "hdfs://172.16.0.105/user/kai/zcg/data/synthetic_sparse_10w_846.csv"
spark = OrcaContext.get_spark_session()
df = spark.read.option("inferSchema", "true").option('header', 'true').csv(data_filepath)
dim = 846
assembler = VectorAssembler(inputCols=df.columns[:dim], outputCol="features")
df = assembler.transform(df)

from pyspark.sql import functions as F
sparse_to_dense_udf = F.udf(lambda x: DenseVector(x), VectorUDT())
df = df.withColumn("features", sparse_to_dense_udf(F.col("features")))
df = df.withColumnRenamed("output", "label")
train, test = df.randomSplit([0.8, 0.2], seed=24)

params = {"tree_method": 'hist', "eta": 0.1, "gamma": 0.1,
          "min_child_weight": 30, "reg_lambda": 1, "scale_pos_weight": 2,
          "subsample": 1, "objective": "binary:logistic"}
# train
import os
import tempfile
import uuid

from bigdl.dllib.utils.file_utils import is_local_path, append_suffix
from bigdl.orca.data.file import put_local_file_to_remote, get_remote_file_to_local


def save_model(model, path):
    if is_local_path(path):
        model.saveModel(path)
    else:
        file_name = str(uuid.uuid1())
        file_name = append_suffix(file_name, path)
        temp_path = os.path.join(tempfile.gettempdir(), file_name)
        try:
            model.saveModel(temp_path)
            put_local_file_to_remote(temp_path, path)
        finally:
            os.remove(temp_path)

def load_model(path, class_num):
    model = None
    if is_local_path(path):
        model = XGBClassifierModel.loadModel(path, class_num)
    else:
        file_name = str(uuid.uuid1())
        file_name = append_suffix(file_name, path)
        temp_path = os.path.join(tempfile.gettempdir(), file_name)
        get_remote_file_to_local(path, temp_path)
        try:
            model = XGBClassifierModel.loadModel(temp_path, class_num)
        finally:
            os.remove(temp_path)
    return model

for eta in [0.1]:
    for max_depth in [6]:
        for num_round in [200]:
            params.update({"eta": eta, "max_depth": max_depth, "num_round": num_round})
            classifier = XGBClassifier(params)
            xgbmodel = classifier.fit(train)
            save_model(xgbmodel, "hdfs:///user/kai/zcg/model/xgb.model")
            xgbmodel = load_model("hdfs:///user/kai/zcg/model/xgb.model", 2)
            xgbmodel.setFeaturesCol("features")
            predicts = xgbmodel.transform(test).drop("features")
            predicts.cache()
            predicts.show(5, False)
            predicts.unpersist(blocking=True)

stop_orca_context()
