import os
import csv
from config import conf

os.chdir(conf["WORKPATH"])
dataset_path = conf['dataset_path']
trainData = conf['trainData']
with open(os.path.join(dataset_path, trainData), 'r', encoding='utf-8') as csvfile:
    cs = list(csv.reader(csvfile))
userId = set()
movieId = set()
value = set()
for record in cs:
    if int(record[0]) not in userId:
        userId.add(int(record[0]))
    if int(record[1]) not in movieId:
        movieId.add(int(record[1]))
    if int(record[2]) not in value:
        value.add(int(record[2]))
    # if int(record[2]) == 0:
        print(record)
# print(userId, len(userId))
print(len(userId)) # 2173
# print(movieId, len(movieId))
print(len(movieId)) # 58431
print(value)