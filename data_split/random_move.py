import random
import os
import shutil

def random_copyfile(srcPath,dstPath,numfiles):
    name_list=list(os.path.join(srcPath,name) for name in os.listdir(srcPath))
    random_name_list=list(random.sample(name_list,numfiles))
    if not os.path.exists(dstPath):
        os.mkdir(dstPath)
    for oldname in random_name_list:
        shutil.move(oldname,oldname.replace(srcPath, dstPath))

srcPath= r'D:\ABC数据集\数据划分\task3\lp3\22_design\total_22'
dstPath = r'D:\ABC数据集\数据划分\task3\lp3\22_design\test'
random_copyfile(srcPath,dstPath, 6600)

