import json
import csv
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer,VectorAssembler,StandardScaler
from pyspark.sql.types import StructType, StructField, IntegerType

from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
from bigdl.orca.automl import hp
from bigdl.orca.automl.xgboost import AutoXGBRegressor, AutoXGBClassifier


label_cols = ["output"]
feature_cols = []
dim = 846
for i in range(dim):
  feature_cols.append('f-' + str(i))

extra_params = {"min-worker-port": "30000", "max-worker-port": "33333", "metrics-export-port": "20010", "include-dashboard": False}
sc = init_orca_context(cluster_mode="spark-submit", cores=3, init_ray_on_spark=True, extra_params=extra_params)

# Load data
#data_filepath = "hdfs://172.16.0.105/user/kai/zcg/data/multi_task_small.csv"
data_filepath = "hdfs://172.16.0.105/user/kai/zcg/data/multi_task.csv"

spark = OrcaContext.get_spark_session()

df = spark.read.option("inferSchema", "true").option('header', 'true').csv(data_filepath)

splits = df.randomSplit([0.8, 0.2], seed=24)
train_df = splits[0]
test_df = splits[1]

search_space = {
    "n_estimators": hp.grid_search([50, 100, 200]),
    "max_depth": hp.choice([2, 4, 6]),
}


auto_xgb_clf = AutoXGBClassifier(cpus_per_trial=1,
                                name="auto_xgb_classifier",
                                min_child_weight=3,
                                random_state=2)

auto_xgb_clf.fit(data=train_df,
                 validation_data=test_df,
                 search_space=search_space,
                 n_sampling=2,
                 metric="error",
                 metric_mode="min",
                 feature_cols=feature_cols,
                 label_cols=label_cols)

best_model = auto_xgb_clf.get_best_model()
best_config = auto_xgb_clf.get_best_config()

print(best_config)

#best_model.save("hdfs://dw-prod-new/das/coe/hsh/e092315/models/xgboost.pkl")
stop_orca_context()
