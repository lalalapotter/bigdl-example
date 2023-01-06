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

import os
import tempfile
import uuid

from bigdl.dllib.utils.file_utils import is_local_path
from bigdl.dllib.utils.file_utils import append_suffix
from bigdl.orca.data.file import put_local_file_to_remote

def save_model(model, path):
    if is_local_path(path):
        model.saveModel(path)
    else:
        file_name = str(uuid.uuid1())
        file_name = append_suffix(file_name, path)
        temp_path = os.path.join(tempfile.gettempdir(), file_name)
        try:
            model.save_model(temp_path)
            put_local_file_to_remote(temp_path, path)
        finally:
            os.remove(temp_path)

save_model(best_model, "hdfs:///user/kai/zcg/model/xgb_automl.model")
stop_orca_context()
