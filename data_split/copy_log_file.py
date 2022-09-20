import os
import shutil
import time
import glob

from joblib import Parallel, delayed

delimiter = "\n"

train_design = ['bc0', 'mainpla', 'div', 'log2', 'cavlc', 'router', 'usb_phy', 'fir', 'fpu', 'tv80']
valid_design = ['apex1', 'i2c', 'aes_xcrypt']
test_design = ['c7552', 'k2', 'sqrt', 'multiplier', 'priority', 'aes', 'pci']
designs = train_design + valid_design + test_design


start_time = time.time()

root_path = "/home/yangch/data_gen/datagen/automation/OPENABC_DATASET/bench/"
backup_log_file = "/home/yangch/data_gen/datagen/automation/OPENABC_DATASET/backup_log_file/"



def cp_log_file(desName,root_path):

    source_path = os.path.join(root_path,desName,'log_'+desName)

    target_path = os.path.join(backup_log_file,'log_'+desName)

    shutil.copytree(source_path, target_path)


# def move_valid_file(desName,id):
#     orgfile = os.path.join(old_valid_path,desName + '_syn{}.pt'.format(id))
#     tar_valid_path = os.path.join(move_valid_path,desName)
#     shutil.move(orgfile, tar_valid_path)

#all_list = os.listdir(old_path)
Parallel(n_jobs=30)(delayed(cp_log_file)(des, root_path) for des in designs)
    #Parallel(n_jobs=20)(delayed(move_train_file)(des, id) for id in range(200000, 215000))



# for des in valid_design:
#
#     Parallel(n_jobs=20)(delayed(move_valid_file)(des, id) for id in range(60000, 70000))
#     Parallel(n_jobs=20)(delayed(move_valid_file)(des, id) for id in range(215000, 215000+2500))

# for des in test_design:
#     file_path = glob.glob(os.path.join(old_path,"test",des,"processed","*.zip"))
#     #print(file_path)
#     for file in file_path:
#         shutil.copy(file, test_path)


end_time = time.time()
print(end_time - start_time, '秒')