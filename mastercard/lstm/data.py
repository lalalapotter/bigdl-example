import csv
import random
from tqdm import trange
from pyspark.sql.types import StructType, StructField, FloatType
from pyspark.sql import SparkSession

records=10000
dim = 128
print("Making %d records\n" % records)
spark = SparkSession.builder.master("local[32]") \
    .appName("Data Generation").getOrCreate()

fieldnames = []
for i in range(dim):
  fieldnames.append('f-' + str(i))

fieldnames.append('output')

def map_func(x):
  row = []
  for f in fieldnames:
      row.append(random.random())
  return row

fields = []
for f in fieldnames:
  fields.append(StructField(f, FloatType(), False))

sc = spark.sparkContext
rdd = sc.range(0, records).map(map_func)
schema = StructType(fields)
data = spark.createDataFrame(rdd, schema)

data.write.mode('overwrite').option('header', 'true').csv("./data.csv")
