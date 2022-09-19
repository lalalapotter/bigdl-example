from pyspark.ml.feature import StringIndexer,VectorAssembler,StandardScaler
from pyspark.sql.types import StructType, StructField, IntegerType
import tensorflow as tf

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca import OrcaContext
from bigdl.orca.learn.tf2 import Estimator

# Orca
OrcaContext.barrier_mode = False
# sc = init_orca_context(cluster_mode="spark-submit", conf={"spark.dynamicAllocation.maxExecutors": "2", "spark.dynamicAllocation.minExecutors": "2"}, num_ray_nodes=2, ray_node_cpu_cores=4)

sc = init_orca_context(cluster_mode="spark-submit")

data_filepath = "hdfs://172.16.0.105/user/kai/zcg/data/large.csv"

spark=OrcaContext.get_spark_session()

fields = []
for i in range(846):
  fields.append(StructField('f-' + str(i), IntegerType(), False))
fields.append(StructField('label', IntegerType(), False))
schema = StructType(fields)

df = spark.read.schema(schema).csv(data_filepath)

assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="features")
df = assembler.transform(df)
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
df = scaler.fit(df).transform(df)

train, test = df.randomSplit([0.8, 0.2], 24)

train_rows = train.count()
test_rows = test.count()


def get_model(config):
    input_layer = tf.keras.layers.Input(shape=(846), name="input")
    x = tf.keras.layers.Dense(units=256, activation="relu", name="dense_1")(input_layer)
    x = tf.keras.layers.Dense(units=256, activation="relu", name="dense_2")(x)
    x = tf.keras.layers.Dense(units=256, activation="relu", name="dense_3")(x)
    x = tf.keras.layers.Dense(units=128, activation="relu", name="dense_4")(x)
    x = tf.keras.layers.Dense(units=64, activation="relu", name="dense_5")(x)
    x = tf.keras.layers.Dense(units=32, activation="relu", name="dense_6")(x)

    output = tf.keras.layers.Dense(units=1, name="output_layer")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output, name="test_model")

    model.compile(optimizer="Adam",
                 loss=tf.keras.losses.mse,
                 metrics=["mae"])
    return model

batch_size=16000

est = Estimator.from_keras(model_creator=get_model, backend="ray", model_dir="hdfs://172.16.0.105:8020/user/kai/zcg/", workers_per_node=2)
est.fit(data=train,
        batch_size=batch_size,
        epochs=10,
        feature_cols=["scaled_features"],
        label_cols=["label"],
        steps_per_epoch=train_rows // batch_size)

est.save("hdfs://172.16.0.105:8020/user/kai/zcg/model.h5")

# tf_model = est.get_model()

# tf_model.save("hdfs://ads-stage-new/user/e092315/model.h5")

est.shutdown()

