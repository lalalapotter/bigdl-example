mlflow server \
    --artifacts-destination hdfs://172.16.0.105:8020/user/kai/zcg/MlflowOutput \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 56789
