import os
import shutil
import time
import glob
from joblib import Parallel, delayed

delimiter = "\n"

train_design = ['bc0', 'mainpla', 'log2', 'cavlc', 'router', 'fir', 'fpu', 'tv80']
valid_design = ['i2c']
#test_design = ['c7552', 'k2', 'sqrt', 'multiplier', 'priority', 'aes', 'pci']


start_time = time.time()

old_path = "/home/yangch/TODAES/data_set/ptdata/"
train_path = "/home/yangch/TODAES/data_set/train/"
valid_path = "/home/yangch/TODAES/data_set/valid/"
test_path = "/home/yangch/TODAES/data_set/test/processed/"

def movefile(org_file, tar_file):

    shutil.move(org_file, tar_file)


all_list = os.listdir(old_path)
for des in train_design:
    file_path = glob.glob(os.path.join(old_path,"train",des,"*.pt"))

    Parallel(n_jobs=20)(delayed(movefile)(file, train_path) for file in file_path)


for des in valid_design:
    file_path = glob.glob(os.path.join(old_path,"valid",des,"*.pt"))
    Parallel(n_jobs=20)(delayed(movefile)(file, valid_path) for file in file_path)
    #print(file_path)

# for des in test_design:
#     file_path = glob.glob(os.path.join(old_path,"test",des,"processed","*.zip"))
#     #print(file_path)
#     for file in file_path:
#         shutil.copy(file, test_path)


end_time = time.time()
print(end_time - start_time, '秒')