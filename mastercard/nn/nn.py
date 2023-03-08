import random

import tensorflow as tf
from pyspark.ml.feature import StringIndexer,VectorAssembler,StandardScaler
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

class NN:

    def __init__(self,
                 model_name,
                 model_path,
                 model_template,
                 feature_columns,
                 label_columns,
                 batch_size,
                 epochs):

        self.model_name = model_name
        self.model_path = model_path
        self.model_template = model_template
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.batch_size = batch_size
        self.epochs = epochs

        self.inferred_df = None

    def generate_classification_dataset(self, records, dim):
        # generate synthetic dataset for binary classification
        print("Making %d records\n" % records)
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

        rdd =self.sc.range(0, records).map(map_func)
        schema = StructType(fields)
        df = self.spark.createDataFrame(rdd, schema)

        # process generated dataframe
        # assembler for input_3
        assembler_input_3 = VectorAssembler(inputCols=df.columns[:dim], outputCol="input_3")
        df = assembler_input_3.transform(df)
        #assembler for decoder_4
        assembler_decoder_4 = VectorAssembler(inputCols=df.columns[dim:2*dim], outputCol="decoder_4")
        df = assembler_decoder_4.transform(df)
        train, test = df.randomSplit([0.8, 0.2], 24)
        return train, test

    def generate_regression_dataset(self, records, dim):
        # generate synthetic dataset for regression
        print("Making %d records\n" % records)
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

        rdd = self.sc.range(0, records).map(map_func)
        schema = StructType(fields)
        df = self.spark.createDataFrame(rdd, schema)

        # process generated dataframe
        # assembler for input_3
        assembler_input_3 = VectorAssembler(inputCols=df.columns[:dim], outputCol="input_3")
        df = assembler_input_3.transform(df)
        return df


    def _init_orca(self):

        from bigdl.orca import init_orca_context, stop_orca_context
        from bigdl.orca import OrcaContext

        OrcaContext.barrier_mode = False
        extra_params = {"min-worker-port": "30000", "max-worker-port": "33333", "metrics-export-port": "20010",
                        "include-dashboard": False}
        self.sc = init_orca_context(cluster_mode="spark-submit", extra_params=extra_params, object_store_memory="30g")

        self.spark = OrcaContext.get_spark_session()

        return self

    def _load_data(self):
        self.train_df, self.validation_df = self.generate_classification_dataset(10000, 846)
        self.inference_df = self.validation_df
        return self

    def _model_generator(self):
        import tensorflow as tf

        def get_model(batch_size, train_data_size, bigdl=False):
            from bigdl.orca.data.file import enable_multi_fs_load_static
            import os
            @enable_multi_fs_load_static
            def read_model(path):
                model_name = path.strip("/").split("/")[-1].split(".")[-1]
                model = tf.keras.models.load_model(os.path.join(path, model_name))
                # print(model.summary())
                return model

            def model_creator(config):
                model = read_model(config["model_path"])
                return model

            if bigdl:
                return model_creator

            return model_creator(None)

        self.model = get_model(self.batch_size, self.num_rows, bigdl=True)

        return self

    def _init_est(self, model_path, backend):
        from bigdl.orca.learn.tf2 import Estimator

        self._model_generator()
        self.est = Estimator.from_keras(model_creator=self.model, config=dict(model_path=model_path),
                                        workers_per_node=2, backend=backend)
        return self

    def _save_model(self, path):
        assert hasattr(self, "est"), "no estimator object to save model"
        self.est.save(path)
        return self

    def run_training(self):
        self._init_orca()
        self._load_data()
        self.num_rows = self.train_df.count()
        assert self.train_df is not None, "training dataframe should not be None"
        self._init_est(model_path=self.model_template, backend="ray")
        self.est.fit(data=self.train_df,
                     batch_size=self.batch_size,
                     epochs=self.epochs,
                     feature_cols=self.feature_columns,
                     label_cols=self.label_columns,
                     steps_per_epoch=self.num_rows // self.batch_size,
                     validation_data=self.validation_df,
                     validation_steps=self.validation_df.count() // self.batch_size)
        self._save_model(self.model_path + self.model_name)

        return self

    def run_inference(self):
        self._init_orca()
        self._load_data()
        self._init_est(model_path=self.model_path+self.model_name, backend="spark")
        self.inferred_df = self.est.predict(self.inference_df, batch_size=self.batch_size,
                                            feature_cols=self.feature_columns)
        print("[INFERENCE LOG] inferred_df:", self.inferred_df.count())
        self.inferred_df.show(5)
        return self