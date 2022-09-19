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
  
4. Submit application.
  ```shell
  ./spark-submit.sh
  ```
