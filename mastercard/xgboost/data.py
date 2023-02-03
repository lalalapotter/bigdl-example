import csv
import random
from tqdm import trange
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.sql import SparkSession

records=100000
dim = 846
print("Making %d records\n" % records)
spark = SparkSession.builder.master("local[32]") \
    .appName("Data Generation").getOrCreate()

fieldnames = []
fields_dense = []
dense_index = random.sample(range(0, dim), 400)
for i in range(dim):
  if i in dense_index:
    fields_dense.append('f-' + str(i))
  fieldnames.append('f-' + str(i))

fieldnames.append('output')

def map_func(x):
  row = []
  for f in fieldnames:
    if f == 'output':
      row.append(random.randint(0, 1))
    elif f in fields_dense and x % 3:
      row.append(0.0)
    else:
      row.append(random.random())
  return row

fields = []
for f in fieldnames:
  if f == 'output':
    fields.append(StructField(f, IntegerType(), False))
  else:
    fields.append(StructField(f, FloatType(), False))


sc = spark.sparkContext
rdd = sc.range(0, records).map(map_func)
schema = StructType(fields)
data = spark.createDataFrame(rdd, schema)

data.write.mode('overwrite').option('header', 'true').csv("/user/kai/zcg/data/synthetic_sparse_10w_864.csv")
