import os
import shutil
import time

import pandas as pd
import networkx as nx
import glob
import pickle
import copy

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.dlpack import to_dlpack, from_dlpack
import scipy.sparse
import zipfile
import argparse

import torch_geometric
import torch_geometric.data
import os.path as osp
from torch_geometric.data import Dataset, download_url
import sys

from joblib import Parallel, delayed


num_trainSynthesizedScript = 60000
#num_validSynthesizedScript = 10000
#num_testSynthesizedScript = 130000

#train_design = ['bc0', 'mainpla', 'div', 'log2', 'cavlc', 'router', 'usb_phy', 'fir', 'fpu', 'tv80']
test_design = ['k2', 'sqrt', 'multiplier', 'priority']
#test_design = ['c7552']


def zipProcessedFolder(path):
    cmd = 'zip -r -q -j '+path+'.zip '+path+os.sep
    os.system(cmd)


def unzipGraphmlFiles(srcZippedFile,destZippedFolder):
    cmd = 'unzip -q '+srcZippedFile+" -d "+destZippedFolder
    os.system(cmd)


def preprocessGraphData(torch_graph,desName, id):

    destFilePath = os.path.join(torch_graph, desName, 'processed', desName + '_syn{}.pt'.format(id))
    #data = torch.load(aig_graphFiles)
    #synPart,stepID = os.path.basename(graphMLFilePath).split('.bench')[0].split('_step')
    #stepID = int(stepID)
    #synID = int(synPart.split('_syn')[1])
    #synID = id
    #data.desName = [desName]
    #data.synVec = torch.tensor(synthID2VecDict[synID])
    #data.synID = [synID]
    #data.stepID = [stepID]
    #destFilePath = osp.join(processedDir, desName + '_syn{}.pt'.format(synID))
    #torch.save(data, destFilePath)
    #zipped_path = os.path.join(zipfile_path, desName, 'processed', desName + '_syn{}.pt'.format(id))
    with zipfile.ZipFile(destFilePath+".zip",'w',zipfile.ZIP_DEFLATED) as fzip:
        fzip.write(destFilePath,arcname=osp.basename(destFilePath))
    os.remove(destFilePath)
    #fzip.close()


def main():
    start_time = time.time()
    #zipfile_path = "/home/yangch/TODAES/data_set/zipfile/train/"
    #synID2Vec = "/home/yangch/gen_test/datagen/utilities/synthID2Vec.pickle"
    torch_graph = "/home/yangch/TODAES/data_set/ptdata/test/"
    for des in test_design:
        design_name = des
        #design_root = os.path.join("/home/yangch/gen_test/datagen/utilities/ptdata/train/", des)

        Parallel(n_jobs=10)(delayed(preprocessGraphData)(torch_graph,design_name,id) for id in range(70000, 200000))
        Parallel(n_jobs=10)(delayed(preprocessGraphData)(torch_graph, design_name, id) for id in range(217500, 250000))

        #rawFolder = os.path.join(sys.argv[1], 'raw')
        #shutil.rmtree(rawFolder)

    end_time = time.time()
    print('total time: ', end_time - start_time)

if __name__ == '__main__':
    main()