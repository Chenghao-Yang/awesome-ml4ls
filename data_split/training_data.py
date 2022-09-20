import os
import shutil
import time
import glob

delimiter = "\n"

train_design = ['bc0', 'mainpla', 'div', 'log2', 'cavlc', 'router', 'usb_phy', 'fir', 'fpu', 'tv80']
valid_design = ['apex1', 'i2c', 'aes_xcrypt']
test_design = ['c7552', 'k2', 'sqrt', 'multiplier', 'priority', 'aes', 'pci']


start_time = time.time()

old_path = "/home/yangch/gen_test/datagen/utilities/ptdata/"
train_path = "/home/yangch/TODAES/data_set/train/processed/"
valid_path = "/home/yangch/TODAES/data_set/valid/processed/"
test_path = "/home/yangch/TODAES/data_set/test/processed/"

all_list = os.listdir(old_path)
for des in train_design:
    file_path = glob.glob(os.path.join(old_path,"train",des,"processed","*.zip"))
    for file in file_path:
    #print(file_path)
        shutil.copy(file, train_path)

for des in valid_design:
    file_path = glob.glob(os.path.join(old_path,"valid",des,"processed","*.zip"))
    #print(file_path)
    for file in file_path:
        shutil.copy(file, valid_path)

for des in test_design:
    file_path = glob.glob(os.path.join(old_path,"test",des,"processed","*.zip"))
    #print(file_path)
    for file in file_path:
        shutil.copy(file, test_path)


end_time = time.time()
print(end_time - start_time, '秒')