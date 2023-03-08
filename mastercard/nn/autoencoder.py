from nn import NN

if __name__ == "__main__":
    autoencoder = NN(model_name="saved_model_new",
                     model_path="hdfs:///user/kai/zcg/model",
                     model_template="hdfs:///user/kai/zcg/model/saved_model",
                     feature_columns=["input_1", "input_2", "input_3"],
                     label_columns=["decoder_4", "output"],
                     batch_size=32,
                     epochs=1)

    autoencoder.run_training()
    autoencoder.run_inference()