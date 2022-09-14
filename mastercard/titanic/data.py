import csv
import random

records=1000000
print("Making %d records\n" % records)

fieldnames=['id']

for i in range(845):
  fieldnames.append('f-' + str(i))

fieldnames.append('label')
print(fieldnames[300:320])

writer = csv.DictWriter(open("large.csv", "w"), fieldnames=fieldnames)

writer.writerow(dict(zip(fieldnames, fieldnames)))
for i in range(0, records):
  row = {}
  for f in fieldnames:
    if f == 'id':
      row[f] = i
    elif f == 'label':
      row[f] = str(random.randint(0, 1))
    else:
      row[f] = str(random.randint(0,100))
  writer.writerow(row)

