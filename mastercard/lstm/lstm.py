from pyspark.ml.feature import VectorAssembler
from bigdl.orca import init_orca_context, OrcaContext
from bigdl.orca.learn.tf2 import Estimator
import tensorflow as tf

sc = init_orca_context(cluster_mode="local", init_ray_on_spark=True)
spark = OrcaContext.get_spark_session()
data_filepath = "./data.csv"
dim = 128
df = spark.read.option("inferSchema", "true").option('header', 'true').csv(data_filepath)
assembler = VectorAssembler(inputCols=df.columns[:dim], outputCol="input")
df = assembler.transform(df)
df.printSchema()
df.show(5)
# design network
def model_creator(config):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=dim, output_dim=64))
    model.add(tf.keras.layers.LSTM(50))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model

estimator = Estimator.from_keras(model_creator=model_creator, backend="spark", worker_per_node=2)
epochs = 1
batch_size = 32
# fit network
estimator.fit(data=df,
              epochs=epochs,
              batch_size=batch_size,
              feature_cols=["input"],
              label_cols=["output"],
              steps_per_epoch= df.count() // batch_size)

# make a prediction

estimator.shutdown()