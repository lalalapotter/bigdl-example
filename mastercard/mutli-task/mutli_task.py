from pyspark.sql import SparkSession
import tensorflow as tf

from pyspark.ml.feature import StringIndexer,VectorAssembler,StandardScaler
from pyspark.sql.types import StructType, StructField, IntegerType

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca import OrcaContext
from bigdl.orca.learn.tf2 import Estimator

import random

# Orca
sc = init_orca_context(cluster_mode="spark-submit")

data_filepath = "hdfs://172.16.0.105/user/kai/zcg/data/large.csv"


def concat(df1, df2):
    from pyspark.sql.functions import monotonically_increasing_id as mi
    id = mi()
    df1 = df1.withColumn("match_id", id)
    df2 = df2.withColumn("match_id", id)
    return df2.join(df1, df1.match_id==df2.match_id, 'inner').drop(df1.match_id)

#Orca
spark=OrcaContext.get_spark_session()

fields = []
for i in range(846):
  fields.append(StructField('f-' + str(i), IntegerType(), False))
fields.append(StructField('output', IntegerType(), False))
schema = StructType(fields)

df = spark.read.schema(schema).csv(data_filepath)
#generate input_3
assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="features")
df = assembler.transform(df)
scaler = StandardScaler(inputCol="features", outputCol="input_3", withStd=True, withMean=True)
train = scaler.fit(df).transform(df)

# generate decoder_4
decoder = train.withColumnRenamed("input_3", "decoder_4")

train_rows = train.count()

# generate input_1
rdd = sc.range(0, train_rows).map(
    lambda x: [random.randint(0, 4)])
schema = StructType([StructField("input_1", IntegerType(), False)])
input_1 = spark.createDataFrame(rdd, schema)

# generate input_2
rdd = sc.range(0, train_rows).map(
    lambda x: [random.randint(0, 4)])
schema = StructType([StructField("input_2", IntegerType(), False)])
input_2 = spark.createDataFrame(rdd, schema)

# concat
train = concat(train, input_1)
train = concat(train, input_2)
train = concat(train, decoder)

batch_size = 16000

def model_creator(config):
    """Stream 1: latest_dw_product_cd"""
    input_1 = tf.keras.Input(shape=(1,), name='input_1')
    embedding_layer_input_1 = tf.keras.layers.Embedding(250, 32, input_length=1)(input_1) # assuming 250 cards
    flatten_layer_input_1 = tf.keras.layers.Flatten()(embedding_layer_input_1)
    s1_dense_1 = tf.keras.layers.Dense(8, activation='relu')(flatten_layer_input_1)
    s1_dense_2 = tf.keras.layers.Dense(16, name='s1_dense_2', activation='relu')(s1_dense_1)

    """Stream 2: dw_iss_country_cd"""
    input_2 = tf.keras.Input(shape=(1,), name='input_2')
    embedding_layer_input_2 = tf.keras.layers.Embedding(250, 32, input_length=1)(input_2) # assuming 250 sovereign states
    flatten_layer_input_2 = tf.keras.layers.Flatten()(embedding_layer_input_2)
    s2_dense_1 = tf.keras.layers.Dense(8, activation='relu')(flatten_layer_input_2)
    s2_dense_2 = tf.keras.layers.Dense(16, name='s2_dense_2', activation='relu')(s2_dense_1)

    """Autoencoder"""
    input_3 = tf.keras.Input(shape=(846,), name='input_3')
    encoder_1 = tf.keras.layers.Dense(512, name='encoder_1', activation='relu')(input_3)
    encoder_2 = tf.keras.layers.Dense(256, name='encoder_2', activation='relu')(encoder_1)
    encoder_3 = tf.keras.layers.Dense(128, name='encoder_3', activation='relu')(encoder_2)

    decoder_1 = tf.keras.layers.Dense(128, name='decoder_1', activation='relu')(encoder_3)
    decoder_2 = tf.keras.layers.Dense(256, name='decoder_2', activation='relu')(decoder_1)
    decoder_3 = tf.keras.layers.Dense(512, name='decoder_3', activation='relu')(decoder_2)
    decoder_4 = tf.keras.layers.Dense(846, name='decoder_4', activation='relu')(decoder_3)

    """Concat Layer"""
    concat_layer = tf.keras.layers.Concatenate(axis=1)([s1_dense_2, s2_dense_2,
                                                        encoder_3])

    """DNN 3"""
    dense_3_1 = tf.keras.layers.Dense(128, name='dense_3_1', activation='relu')(concat_layer)
    dense_3_2 = tf.keras.layers.Dense(64, name='dense_3_2', activation='relu')(dense_3_1)
    dense_3_3 = tf.keras.layers.Dense(32, name='dense_3_3', activation='relu')(dense_3_2)
    dense_3_3 = tf.keras.layers.BatchNormalization()(dense_3_3)
    dense_3_3 = tf.keras.layers.Dropout(0.2)(dense_3_3)

    output = tf.keras.layers.Dense(2, name='output', activation='softmax')(dense_3_3)

    model = tf.keras.Model(inputs=[input_1, input_2, input_3],
                           outputs=[decoder_4, output])
    # num_steps = num of records / batch size
    num_steps = int(train_rows / batch_size)

    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [100 * num_steps, 100 * num_steps, 100 * num_steps, 100 * num_steps, 100 * num_steps, 100 * num_steps,
         100 * num_steps, 100 * num_steps, 100 * num_steps, 100 * num_steps, 100 * num_steps],
        [1e-3, 1e-3, 1e-2, 1e-1, 1e-3, 1e-3, 1e-2, 1e-4, 1e-3, 1e-2, 1e-3, 1e-4]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

    model.compile(
        optimizer=optimizer,
        loss={
            "decoder_4": "mse",
            "output": "sparse_categorical_crossentropy",
        },
        metrics=["accuracy"]
    )

    return model


# Orca
est = Estimator.from_keras(model_creator=model_creator, backend="ray", model_dir="hdfs://172.16.0.105:8020/user/kai/zcg/", workers_per_node=2)
est.fit(data=train,
        batch_size=batch_size,
        epochs=1,
        feature_cols=["input_1", "input_2", "input_3"],
        label_cols=["decoder_4", "output"],
        steps_per_epoch=train_rows // batch_size)

est.shutdown()

