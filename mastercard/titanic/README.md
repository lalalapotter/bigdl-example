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

    You should change the path in data.py to your own HDFS path, and replace the HDFS path in titanic_large.py as well. Note that the model save path in titanic_large.py also should be replace to your own HDFS path.
  
    ```shell
    python data.py
    ```
  
5. Submit application.
 
    ```shell
    ./spark-submit.sh
    ```

Note that, if you are using bigdl-2.0, please use `tf2` backend instead of `ray` and use the bigdl-2.0 dependencies. Other than that, you don't need to modify any code.
