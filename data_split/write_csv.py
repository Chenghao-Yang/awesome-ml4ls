import os
import csv
import time

start_time = time.time()

train_path = "/home/yangch/TODAES/data_set/train/processed/"
valid_path = "/home/yangch/TODAES/data_set/valid/processed/"
test_path = "/home/yangch/TODAES/data_set/test/processed/"

l_train=[]
l_valid=[]
l_test=[]

for trainfile in os.listdir(train_path):
    l_train.append(trainfile)
traincsv = open("train_data.csv","a")
traincsv.write('fileName' + '\n')
for line in l_train:
    traincsv.write(line+'\n')
traincsv.close()


for validfile in os.listdir(valid_path):
    l_valid.append(validfile)
validcsv = open("valid_data.csv","a")
validcsv.write('fileName' + '\n')
for line in l_valid:
    validcsv.write(line+'\n')
validcsv.close()


for testfile in os.listdir(test_path):
    l_test.append(testfile)
testcsv = open("test_data.csv","a")
testcsv.write('fileName' + '\n')
for line in l_test:
    testcsv.write(line+'\n')
testcsv.close()

end_time = time.time()
print(end_time - start_time, '秒')