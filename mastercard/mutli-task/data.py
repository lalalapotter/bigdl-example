import csv
import random

from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql import SparkSession

records=10000000
dim = 846
print("Generating %d records\n" % records)
spark = SparkSession.builder.master("local[32]") \
    .appName("Data Generation").getOrCreate()

fieldnames = []
for i in range(dim):
  fieldnames.append('f-' + str(i))
fieldnames.append('label')

def map_func(x):
  row = []
  for f in fieldnames:
    if f == 'label':
      row.append(random.randint(0, 1))
    else:
      row.append(random.randint(0,100))
  return row

fields = []
for f in fieldnames:
  fields.append(StructField(f, IntegerType(), False))

sc = spark.sparkContext
rdd = sc.range(0, records).map(map_func)
schema = StructType(fields)
data = spark.createDataFrame(rdd, schema)
data.write.csv("/user/kai/zcg/data/large.csv")
