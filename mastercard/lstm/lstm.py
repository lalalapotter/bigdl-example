import random

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, FloatType
import tensorflow as tf

from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
from bigdl.orca.learn.tf2 import Estimator


sc = init_orca_context(cluster_mode="spark-submit", init_ray_on_spark=True)

# generate random data
records=10000
dim = 128
print("Generating %d records\n" % records)
spark = OrcaContext.get_spark_session()

fieldnames = []
for i in range(dim):
  fieldnames.append('f-' + str(i))
fieldnames.append('output')

def map_func(x):
  row = []
  for f in fieldnames:
      row.append(random.random())
  return row

fields = []
for f in fieldnames:
  fields.append(StructField(f, FloatType(), False))

rdd = sc.range(0, records).map(map_func)
schema = StructType(fields)
df = spark.createDataFrame(rdd, schema)

assembler = VectorAssembler(inputCols=df.columns[:dim], outputCol="input")
df = assembler.transform(df)
train_df, test_df = df.randomSplit([0.8, 0.2], 24)

# design network
def model_creator(config):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=dim, output_dim=64))
    model.add(tf.keras.layers.LSTM(50))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model

estimator = Estimator.from_keras(model_creator=model_creator, backend="ray", worker_per_node=2)
epochs = 1
batch_size = 32

# fit network
estimator.fit(data=train_df,
              epochs=epochs,
              batch_size=batch_size,
              feature_cols=["input"],
              label_cols=["output"],
              steps_per_epoch= df.count() // batch_size)

# make a prediction
estimator.predict(data=test_df,
                  feature_cols=["input"])

stop_orca_context()