Please follow the steps to run the example.
1. Make directory for dependencies.

    ```shell
    mkdir deps
    ```

2. Download dependencies from [link-for-spark-2.4.6](https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-assembly-spark_2.4.6/2.1.0-SNAPSHOT/bigdl-assembly-spark_2.4.6-2.1.0-20220828.122449-192.zip) and unzip it to deps directory.
  
    ```shell
    cd deps
    wget https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-assembly-spark_2.4.6/2.1.0-SNAPSHOT/bigdl-assembly-spark_2.4.6-2.1.0-20220828.122449-192.zip
    unzip bigdl-assembly-spark_2.4.6-2.1.0-20220828.122449-192.zip
    ```

3. Pack the environment.
  
    ```shell
    conda pack -o environment.tar.gz
    ```

4. Generate random data using `data.py`.

    You should change the path in data.py to your own HDFS path, and replace the HDFS path in multi_task.py as well. Note that the model save path in multi_task.py also should be replace to your own HDFS path.
  
    ```shell
    python data.py
    ```
  
5. Submit application.
 
    ```shell
    ./spark-submit.sh
    ```

6. Monitor application with TensorBoard.

   You should set the environment variable in `tensorboard.sh` according to the instruction of [TensorFlow on Hadoop](https://github.com/tensorflow/examples/blob/tflmm/v0.2.4/community/en/docs/deploy/hadoop.md), and update the HDFS path as well.
   ```
   ./tensorboard.sh
   ```

7. Monitor application with Mlflow.

   You should set the HDFS path in `mlflow-server.sh` and start it before running the application with following command.
   ```
   ./mlflow-server.sh
   ```
Note that, if you are using bigdl-2.0, please use `tf2` backend instead of `ray` and use the bigdl-2.0 dependencies. Other than that, you don't need to modify any code.
