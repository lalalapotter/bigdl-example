import random

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import layers
from tensorflow.python.keras.layers import *
from pyspark.sql.types import StructType, StructField, FloatType
from pyspark.ml.linalg import Vectors, VectorUDT

from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
from bigdl.orca.learn.tf2 import Estimator

def generate_sample_data(num, sc):
    print("Generating %d sample data\n" % num)
    spark = OrcaContext.get_spark_session()
    fieldsname = ["scaled_trx_amt_v","scaled_trx_diff_v","trx_time_hours_v","txn_month_v","txn_weekday_v","product_code_v","cardholder_txn_type_v","city_name_v","merchant_name_v","mcc_code_v","merchant_location_id_v","label_per_client"]

    def map_func(x):
        row = []
        for field in fieldsname:
            if field == "label_per_client":
                row.append(float(random.randint(0, 1)))
            else:
                row.append(Vectors.dense([random.random() for i in range(0, 200)]))
        return row

    fields = []
    for f in fieldsname:
        if f == "label_per_client":
            fields.append(StructField(f, FloatType(), True))
        else:
            fields.append(StructField(f, VectorUDT(), True))

    rdd = sc.range(0, num).map(map_func)
    schema = StructType(fields)
    df = spark.createDataFrame(rdd, schema)
    df.printSchema()
    df.show(5)
    train, test = df.randomSplit([0.8, 0.2], 24)
    return train, test

def get_model(voc_dict, emb_dict, cat_cols, num_cols, HIST_TRX=200, MASK_VALUE=0, bigdl=True):

    def model_creator(config):

        num_inputs = []
        num_masks = []
        for col in num_cols:
            inp_name = col
            mask_name = col+'_mask'
            inp = layers.Input(shape=(HIST_TRX), name=inp_name)
            mask= layers.Masking(mask_value=MASK_VALUE, name=mask_name)(inp)
            num_inputs.append(inp)
            num_masks.append(mask)


        nums = []
        for layer in num_masks:
            layer_reshaped = layers.Reshape((HIST_TRX, 1))(layer)
            nums.append(layer_reshaped)

        num_proc=layers.concatenate(nums, name='num_concatenation')


        cat_inputs = []
        cat_masks = []
        for col in cat_cols:
            #inp, mask = create_cat_inp_layer(col)
            inp_layer = layers.Input(shape=(HIST_TRX), name=col+"_inp")
            mask_layer = tf.keras.layers.Masking(mask_value=MASK_VALUE, name=col+"_mask")(inp_layer)
            cat_inputs.append(inp_layer)
            cat_masks.append(mask_layer)



        embs = []

        for i, layer in enumerate(cat_masks):
            col = cat_cols[i]
            #emb_layer = create_emb_layer(col, layer)
            voc_size=voc_dict[col]
            emb_dim=emb_dict[col]
            emb_layer=layers.Embedding(voc_size, emb_dim, input_length=HIST_TRX, name=col+"_emb")(layer)
            embs.append(emb_layer)
        emb_proc=layers.concatenate(embs, name='embs_concatenation')


    #emb_proc  is the last layer here for cat ts cols


        full_conc=layers.concatenate([emb_proc, num_proc], name= 'full_concatenation')
        gru=layers.LSTM(
            units=200,
            return_sequences=True,
            name="GRU-1",kernel_initializer='he_uniform',
            dropout=0.1
            )(full_conc)




        maxpool = layers.MaxPool1D(pool_size=200, name='maxpool'+"after_GRU")(gru)
        avpool = tf.keras.layers.AveragePooling1D(pool_size=200, name='avpool'+"after_GRU")(gru)
        pool=layers.concatenate([maxpool, avpool], name='pool_concat'+"after_GRU")
        gru2= layers.LSTM(
                100,
                return_sequences=False,
                name="GRU-2",kernel_initializer='he_uniform')(pool)



        gru2_reshaped = layers.Reshape((1, 100))(gru2)
        weird_conc = layers.concatenate([pool, gru2_reshaped], name='weird_concatenation')
        #x = Flatten()(weird_conc)  ### Inputs and layer processing for the Nums and Cats variable
        d1 = layers.Dense(100, activation="relu", name="dense_111",kernel_initializer='he_uniform')(weird_conc)


        x = layers.Flatten()(d1)
        d2=layers.Dense(50, activation="relu", name="predict")(x)
        d3=layers.Dense(1, activation="sigmoid", name="predictions")(d2)


        model = tf.keras.Model(
        inputs=num_inputs+cat_inputs,
        outputs=d3)


        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000988),
                      loss=tf.keras.losses.BinaryFocalCrossentropy(),
                      metrics=[
                        tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name="Precision"),
                        tf.keras.metrics.Recall(name="Recall"),
                        tf.keras.metrics.TruePositives(name="TruePositives"),
                        tf.keras.metrics.TrueNegatives(name="TrueNegatives"),
                        tf.keras.metrics.FalseNegatives(name="FalseNegatives"),
                        tf.keras.metrics.FalsePositives(name="FalsePositives")
                      ]
                    )

        return model

    if bigdl:

        return model_creator

    return model_creator(None)


if __name__ == "__main__":
    sc = init_orca_context(cluster_mode="spark-submit")
    train, test = generate_sample_data(10000, sc)
    voc_dict = {"scaled_trx_amt_v": 200,"scaled_trx_diff_v": 200,"trx_time_hours_v": 200,"txn_month_v": 200,"txn_weekday_v": 200,"product_code_v": 200,"cardholder_txn_type_v": 200,"city_name_v": 200,"merchant_name_v": 200,"mcc_code_v": 200,"merchant_location_id_v": 200}
    emb_dict = {"product_code_v": 4,"cardholder_txn_type_v": 3,"city_name_v": 154,"merchant_name_v": 569,"mcc_code_v": 24,"merchant_location_id_v": 600}
    num_cols = ["scaled_trx_amt_v","scaled_trx_diff_v","trx_time_hours_v","txn_month_v","txn_weekday_v"]
    cat_cols = ["product_code_v", "cardholder_txn_type_v", "city_name_v", "merchant_name_v", "mcc_code_v", "merchant_location_id_v"]
    estimator = Estimator.from_keras(model_creator=get_model(voc_dict, emb_dict, cat_cols, num_cols), backend="spark", worker_per_node=2)
    epochs = 1
    batch_size = 32

    # fit network
    estimator.fit(data=train,
                  epochs=epochs,
                  batch_size=batch_size,
                  feature_cols=["scaled_trx_amt_v","scaled_trx_diff_v","trx_time_hours_v","txn_month_v","txn_weekday_v","product_code_v","cardholder_txn_type_v","city_name_v","merchant_name_v","mcc_code_v","merchant_location_id_v"],
                  label_cols=["label_per_client"],
                  steps_per_epoch= train.count() // batch_size)

    # make a prediction
    estimator.predict(data=test,
                      feature_cols=["scaled_trx_amt_v","scaled_trx_diff_v","trx_time_hours_v","txn_month_v","txn_weekday_v","product_code_v","cardholder_txn_type_v","city_name_v","merchant_name_v","mcc_code_v","merchant_location_id_v"])

    stop_orca_context()
