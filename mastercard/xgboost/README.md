Please follow the steps to run the example.
1. Make directory for dependencies.

    ```shell
    mkdir deps
    ```

2. Download dependencies from [link-for-spark-2.4.6](https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-assembly-spark_2.4.6/2.2.0/bigdl-assembly-spark_2.4.6-2.2.0-fat-jars.zip) and unzip it to deps directory.
  
    ```shell
    cd deps
    wget https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-assembly-spark_2.4.6/2.2.0/bigdl-assembly-spark_2.4.6-2.2.0-fat-jars.zip
    unzip bigdl-assembly-spark_2.4.6-2.2.0-fat-jars.zip
    ```

3. Pack the environment.
  
    ```shell
    conda pack -o environment.tar.gz
    ```

4. Generate random data using `data.py`.

    You should change the path in data.py to your own HDFS path, and replace the HDFS path in xgb.py and xgb_automl.py as well.
  
    ```shell
    python data.py
    ```
  
5. Submit application.
 
    ```shell
    # submit xgb clf application
    ./spark-submit-xgb-clf.sh

    # submit xgb clf automl application
    ./spark-submit-xgb-clf-automl.sh

    # submit xgb reg application
    ./spark-submit-xgb-reg.sh

    # submit xgb reg automl application
    ./spark-submit-xgb-reg-automl.sh
    ```
