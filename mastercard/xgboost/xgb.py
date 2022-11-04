from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer,VectorAssembler,StandardScaler

from bigdl.dllib.nnframes.tree_model import *
from bigdl.dllib.utils.log4Error import *
from bigdl.orca import OrcaContext, init_orca_context, stop_orca_context


if __name__ == '__main__':
    # init orca context
    sc = init_orca_context("spark-submit")
    # load and preprocess dataset
    data_filepath = "hdfs://172.16.0.105/user/kai/zcg/data/multi_task.csv"
    spark = OrcaContext.get_spark_session()
    df = spark.read.option("inferSchema", "true").option('header', 'true').csv(data_filepath)
    dim = 846
    assembler = VectorAssembler(inputCols=df.columns[:dim], outputCol="features")
    df = assembler.transform(df)
    df = df.withColumnRenamed("output", "label")
    train, test = df.randomSplit([0.8, 0.2], seed=24)

    params = {"tree_method": 'hist', "eta": 0.1, "gamma": 0.1,
              "min_child_weight": 30, "reg_lambda": 1, "scale_pos_weight": 2,
              "subsample": 1, "objective": "binary:logistic"}
    # train
    for eta in [0.1]:
        for max_depth in [6]:
            for num_round in [200]:
                params.update({"eta": eta, "max_depth": max_depth, "num_round": num_round})
                classifier = XGBClassifier(params)
                xgbmodel = classifier.fit(train)
                xgbmodel.setFeaturesCol("features")
                predicts = xgbmodel.transform(test).drop("features")
                predicts.cache()
                predicts.show(5, False)

                evaluator = BinaryClassificationEvaluator(labelCol="label",
                                                          rawPredictionCol="rawPrediction")
                auc = evaluator.evaluate(predicts, {evaluator.metricName: "areaUnderROC"})

                evaluator2 = MulticlassClassificationEvaluator(labelCol="label",
                                                               predictionCol="prediction")
                acc = evaluator2.evaluate(predicts, {evaluator2.metricName: "accuracy"})
                print(params)
                print("AUC: %.2f" % (auc * 100.0))
                print("Accuracy: %.2f" % (acc * 100.0))

                predicts.unpersist(blocking=True)

    stop_orca_context()
