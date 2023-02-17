import random

import tensorflow as tf
from pyspark.ml.feature import StringIndexer,VectorAssembler,StandardScaler
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca import OrcaContext
from bigdl.orca.learn.tf2 import Estimator

def generate_classification_dataset(records, dim, sc):
    # generate synthetic dataset for binary classification
    print("Making %d records\n" % records)
    spark = OrcaContext.get_spark_session()
    fieldnames = []
    for i in range(dim):
      fieldnames.append('f-' + str(i))
    for i in range(dim):
      fieldnames.append('d-' + str(i))

    fieldnames.append('output')
    fieldnames.append('input_1')
    fieldnames.append('input_2')

    def map_func(x):
      row = []
      for f in fieldnames:
        if f == 'output':
          row.append(random.randint(0, 1))
        elif f == 'input_1':
          row.append(random.randint(0, 4))
        elif f == 'input_2':
          row.append(random.randint(0, 4))
        else:
          row.append(random.random())
      return row

    fields = []
    for f in fieldnames:
      if f in ['input_1', 'input_2', 'output']:
        fields.append(StructField(f, IntegerType(), False))
      else:
        fields.append(StructField(f, FloatType(), False))

    rdd = sc.range(0, records).map(map_func)
    schema = StructType(fields)
    df = spark.createDataFrame(rdd, schema)

    # process generated dataframe
    # assembler for input_3
    assembler_input_3 = VectorAssembler(inputCols=df.columns[:dim], outputCol="input_3")
    df = assembler_input_3.transform(df)
    #assembler for decoder_4
    assembler_decoder_4 = VectorAssembler(inputCols=df.columns[dim:2*dim], outputCol="decoder_4")
    df = assembler_decoder_4.transform(df)
    train, test = df.randomSplit([0.8, 0.2], 24)
    return train, test

def generate_regression_dataset(records, dim, sc):
    # generate synthetic dataset for regression
    print("Making %d records\n" % records)
    spark = OrcaContext.get_spark_session()
    fieldnames = []
    for i in range(dim):
      fieldnames.append('f-' + str(i))

    fieldnames.append('output')
    fieldnames.append('input_1')
    fieldnames.append('input_2')

    def map_func(x):
      row = []
      for f in fieldnames:
        if f == 'output':
          row.append(random.random()*100.0 + 100.0)
        elif f == 'input_1':
          row.append(random.randint(0, 4))
        elif f == 'input_2':
          row.append(random.randint(0, 4))
        else:
          row.append(random.random())
      return row

    fields = []
    for f in fieldnames:
      if f in ['input_1', 'input_2']:
        fields.append(StructField(f, IntegerType(), False))
      else:
        fields.append(StructField(f, FloatType(), False))

    rdd = sc.range(0, records).map(map_func)
    schema = StructType(fields)
    df = spark.createDataFrame(rdd, schema)

    # process generated dataframe
    # assembler for input_3
    assembler_input_3 = VectorAssembler(inputCols=df.columns[:dim], outputCol="input_3")
    df = assembler_input_3.transform(df)
    train, test = df.randomSplit([0.8, 0.2], 24)
    return train, test

def get_model(batch_size, train_data_size, bigdl=False):
    """
    Define Keras model architecture for evaluation on top of Spark and BigDL in a distributed fashion
    :param batch_size: Batch size for training int
    :param train_data_size: Size of train dataset int
    :param bigdl: Is the model being trained using bigdl?
    :return: Return KerasModel
    """
    def model_creator(config):
        """Stream 1: latest_dw_product_cd"""
        input_1 = tf.keras.Input(shape=(1,), name='input_1')
        embedding_layer_input_1 = tf.keras.layers.Embedding(300, 32, input_length=1)(input_1) # assuming 250 cards
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
        num_steps = int(train_data_size / batch_size)

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

    if bigdl:
        return model_creator

    return model_creator(None)

def get_transfer_learning_model(batch_size, train_data_size, bigdl=False):

    import os
    from bigdl.orca.data.file import enable_multi_fs_load_static
    @enable_multi_fs_load_static
    def read_model(path):
        model_name = path.strip("/").split("/")[-1].split(".")[-1]
        model = tf.keras.models.load_model(os.path.join(path, model_name))
        print(model.summary())
        return model

    def model_creator(config):
        parent_model = read_model(config["model_path"])
        parent_model.trainable = False # Freeze the model

        x = tf.keras.layers.Dense(512, activation="relu", name="d0")(parent_model.get_layer("concatenate").output)
        x = tf.keras.layers.Dense(256, activation="relu", name="d1")(x)
        x = tf.keras.layers.Dense(128, activation="relu", name="d2")(x)
        x = tf.keras.layers.Dense(64, activation="relu", name="d3")(x)
        o = tf.keras.layers.Dense(1, name="output")(x)

        new_model = tf.keras.Model(inputs=[parent_model.get_layer("input_1").output, parent_model.get_layer("input_2").output, parent_model.get_layer("input_3").output], outputs=[o])

        #num_steps = num of records / batch size
        num_steps = int(train_data_size / batch_size)

        learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                [100 * num_steps, 100 * num_steps, 100 * num_steps, 100 * num_steps, 100 * num_steps, 100 * num_steps,
                 100 * num_steps, 100 * num_steps, 100 * num_steps, 100 * num_steps, 100 * num_steps],
                [1e-3, 1e-3, 1e-2, 1e-1, 1e-3, 1e-3, 1e-2, 1e-4, 1e-3, 1e-2, 1e-3, 1e-4]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

        new_model.compile(
                optimizer=optimizer,
                loss="mse",
                metrics=["mae"]
        )

        return new_model

    if bigdl:
        return model_creator

    return model_creator(None)

if __name__ == "__main__":
    sc = init_orca_context(cluster_mode="spark-submit")
    # generate classification dataset
    train, test = generate_classification_dataset(10000, 846, sc)
    train_rows = train.count()
    test_rows = test.count()
    batch_size = 160
    est = Estimator.from_keras(model_creator=get_model(batch_size, train_rows, True), backend="ray")
    # pretrain classification model(you can remove it if you have pretrained model)
    est.fit(data=train,
           batch_size=batch_size,
           epochs=1,
           feature_cols=["input_1", "input_2", "input_3"],
           label_cols=["decoder_4", "output"],
           steps_per_epoch=train_rows // batch_size)

    model_path = "hdfs:///user/kai/zcg/model/saved_model"
    config = {"model_path": model_path}
    est.save(filepath=model_path)
    est.shutdown()

    # generate regression dataset
    train, test = generate_regression_dataset(10000, 846, sc)
    train_rows = train.count()
    test_rows = test.count()
    # init estimator and load model
    est = Estimator.from_keras(model_creator=get_transfer_learning_model(batch_size, train_rows, True), config=config, backend="ray", workers_per_node=2)
    # inference
    result = est.predict(data=test,
                         batch_size=batch_size,
                         feature_cols=["input_1", "input_2", "input_3"])

    for row in result.select(["prediction"]).collect()[:5]:
        print("prediction: ", row[0])

    result.write.save("hdfs:///user/kai/zcg/regression_inference.parquet", format="parquet", mode="overwrite")

    spark = OrcaContext.get_spark_session()
    loaded_result = spark.read.load("hdfs://user/kai/zcg/regression_inference.parquet")
    for row in loaded_result.select(["prediction"]).collect()[:5]:
        print("prediction: ", row[0])

    est.shutdown()