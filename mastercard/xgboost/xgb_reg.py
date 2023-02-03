from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.ml.feature import VectorAssembler

from bigdl.dllib.nnframes.tree_model import *
from bigdl.dllib.utils.log4Error import *
from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext


if __name__ == '__main__':
    sc = init_orca_context("spark-submit")
    data_filepath = "hdfs:///user/kai/zcg/data/synthetic_sparse_10w_846.csv"
    spark=OrcaContext.get_spark_session()
    df = spark.read.option("inferSchema", "true").option('header', 'true').csv(data_filepath)
    dim = 846
    assembler = VectorAssembler(inputCols=df.columns[:dim], outputCol="features").setHandleInvalid("keep")
    df = assembler.transform(df)
    from pyspark.sql import functions as F

    sparse_to_dense_udf = F.udf(lambda x: DenseVector(x), VectorUDT())
    df = df.withColumn("features", sparse_to_dense_udf(F.col("features")))
    df.show(5, False)
    features_col = []
    for i in range(dim):
        features_col.append("f-" + str(i))
    df = df.withColumnRenamed("output", "label")
    train, test = df.randomSplit([0.8, 0.2], seed=24)

    params = {"tree_method": 'hist', "eta": 0.1, "gamma": 0.1,
              "min_child_weight": 30, "reg_lambda": 1, "scale_pos_weight": 2,
              "subsample": 1, "objective": "reg:squarederror"}

    for eta in [0.1]:
        for max_depth in [6]:
            for num_round in [200]:
                params.update({"eta": eta, "max_depth": max_depth, "num_round": num_round})
                regressor = XGBRegressor(params)
                xgbmodel = regressor.fit(train)
                xgbmodel.save("hdfs:///user/kai/zcg/model/xgb_reg.model")
                xgbmodel = XGBRegressorModel.load("hdfs:///user/kai/zcg/model/xgb_reg.model")
                xgbmodel.setFeaturesCol("features")
                predicts = xgbmodel.transform(test).drop("features")
                predicts.cache()
                predicts.show(5, False)
                predicts.unpersist(blocking=True)

    stop_orca_context()