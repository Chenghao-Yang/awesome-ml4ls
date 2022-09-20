import os
import shutil
import time

import pandas as pd
import networkx as nx
import glob
import pickle
import copy
import numpy as np

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


num_trainSynthesizedScript = 15000
#num_validSynthesizedScript = 10000
#num_testSynthesizedScript = 130000

train_design = ['bc0', 'mainpla', 'div', 'log2', 'cavlc', 'router', 'usb_phy', 'fir', 'fpu', 'tv80']
#valid_design = ['apex1', 'i2c', 'aes_xcrypt']
#test_design = ['c7552', 'k2', 'sqrt', 'multiplier', 'priority', 'aes', 'fpu']

IS_STATS_AVAILABLE = True

if IS_STATS_AVAILABLE:
    with open(osp.join("/home/yangch/TODAES/data_set/", 'synthesisStatistics.pickle'), 'rb') as f:
        numGatesAndLPStats = pickle.load(f)
else:
    print("\nNo pickle file found for number of gates")
    exit(0)



def zipProcessedFolder(path):
    cmd = 'zip -r -q -j '+path+'.zip '+path+os.sep
    os.system(cmd)


def unzipGraphmlFiles(srcZippedFile,destZippedFolder):
    cmd = 'unzip -q '+srcZippedFile+" -d "+destZippedFolder
    os.system(cmd)


def getMeanAndVariance(targetList):
    return np.mean(np.array(targetList)),np.std(np.array(targetList))

def computeMeanAndVarianceOfTargets(targetStatsDict,targetVar='nodes'):
    meanAndVarTargetDict = {}
    for des in targetStatsDict.keys():
        #numNodes,_,_,areaVar,delayVar = targetStatsDict[des]
        areaVar, delayVar = targetStatsDict[des]

        delay_meanTarget,delay_varTarget = getMeanAndVariance(delayVar)
        area_meanTarget,area_varTarget = getMeanAndVariance(areaVar)

        meanAndVarTargetDict[des] = [area_meanTarget,area_varTarget,delay_meanTarget,delay_varTarget]
    return meanAndVarTargetDict

meanVarTargetDict = computeMeanAndVarianceOfTargets(numGatesAndLPStats)


def train_addNormalizedTargets(data,targetStatsDict,meanVarDataDict,targetVar='nodes'):
    sid = data.synID[0]
    desName = data.desName[0]
    if sid < 60000:
        sid = sid
    if 199999 < sid < 215000:
        sid = sid - 140000


    delay_targetIdentifier = 1 # Column number of target 'Delay' in synthesisStatistics.pickle entries
    normTarget_delay = (targetStatsDict[desName][delay_targetIdentifier][sid] - meanVarDataDict[desName][2]) / meanVarDataDict[desName][3]
    data.target_delay = torch.tensor([normTarget_delay],dtype=torch.float32)

    area_targetIdentifier = 0 # Column number of target 'Area' in synthesisStatistics.pickle entries
    normTarget_area = (targetStatsDict[desName][area_targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
    data.target_area = torch.tensor([normTarget_area],dtype=torch.float32)

    return data


def preprocessGraphData(aig_graphFiles,processedDir,desName,synthID2VecDict, id):

    data = aig_graphFiles
    #synPart,stepID = os.path.basename(graphMLFilePath).split('.bench')[0].split('_step')
    #stepID = int(stepID)
    #synID = int(synPart.split('_syn')[1])
    synID = id
    data.desName = [desName]
    data.synVec = torch.tensor(synthID2VecDict[synID])
    data.synID = [synID]
    #data.stepID = [stepID]
    data = train_addNormalizedTargets(data, numGatesAndLPStats, meanVarTargetDict)
    destFilePath = osp.join(processedDir, desName + '_syn{}.pt'.format(synID))
    torch.save(data, destFilePath)



class NetlistGraphDataset(Dataset):
    def __init__(self,root,des,design_graph,ptDataLoc,transform=None, pre_transform=None):
        self.des = des
        self.design_graph = design_graph
        self.ptDataLoc = ptDataLoc
        super(NetlistGraphDataset, self).__init__(root,transform,pre_transform)


    # @property
    # def raw_file_names(self):
    #     rawFolder = os.path.join(self.root,'raw')
    #     if not os.path.exists(rawFolder):
    #         zippedGraphmlFolders = [os.path.join(self.graphmlLoc,'syn'+str(i)+'.zip') for i in range(5)]
    #         #origGraphmlFolderZip = os.path.join(os.path.dirname(self.graphmlLoc),'orig',self.des)
    #         os.mkdir(rawFolder)
    #         for synZippedFolder in zippedGraphmlFolders:
    #             unzipGraphmlFiles(synZippedFolder, rawFolder)
    #         #unzipGraphmlFiles(origGraphmlFolderZip,rawFolder)
    #     graphMLfile = glob.glob(rawFolder+os.sep+"*.graphml")
    #     return graphMLfile



    @property
    def processed_file_names(self):
        processedFiles = []
        for i in range(200000, 200000+num_trainSynthesizedScript):
            processedFiles.append(self.des+"_"+ str(i) + '.pt')
        return processedFiles


    def process(self):
        #aig_graphFiles = self.design_graph
        aig_graphFiles = torch.load(self.design_graph)
        synth2vecFile = self.ptDataLoc
        with open(synth2vecFile,'rb') as f:
            synthID2VecDict = pickle.load(f)
        Parallel(n_jobs=10)(delayed(preprocessGraphData)(aig_graphFiles,self.processed_dir,self.des,synthID2VecDict, id) for id in range(200000, 200000+num_trainSynthesizedScript))


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.des+'_{}.pt'.format(idx)))
        return data


def setGlobalAndEnvironmentVars(cmdArgs):
    global homeDir,benchDataFolder,statsDataFolder
    homeDir = cmdArgs.home
    if not (os.path.exists(homeDir)):
        print("\nPlease rerun with appropriate paths")
    benchDataFolder = os.path.join(homeDir,"OPENABC_DATASET","bench")
    statsDataFolder = os.path.join(homeDir,"OPENABC_DATASET","statistics")

def parseCmdLineArgs():
    parser = argparse.ArgumentParser(prog='Graphml 2 pytorch-geometric data conversion', description="Circuit characteristics")
    parser.add_argument('--version',action='version', version='1.0.0')
    parser.add_argument('--des',required=True, help="Design directory (eg. ~/OPENABC_DATASET/ptdata/designName)")
    parser.add_argument('--name', required=True, help="Design name")
    parser.add_argument('--gs', required=True, help="Graphml source directory (eg. ~/OPENABC_DATASET/graphml/designName)")
    parser.add_argument('--synvec', required=True,help="Synthesis vector pickle path (eg. ~/OPENABC_DATASET/statistics/synthID2Vec.pickle)")
    return parser.parse_args()

def main():
    start_time = time.time()

    synID2Vec = "/home/yangch/gen_test/datagen/utilities/synthID2Vec.pickle"
    torch_graph = "/home/yangch/gen_test/datagen/utilities/torch_graph/"
    for des in train_design:
        design_name = des
        design_root = os.path.join("/home/yangch/TODAES/data_set/ptdata/train/", des)
        design_graph = os.path.join(torch_graph, des+".pt")
        aigDataset = NetlistGraphDataset(design_root, design_name, design_graph, synID2Vec)
        #rawFolder = os.path.join(sys.argv[1], 'raw')
        #shutil.rmtree(rawFolder)

    end_time = time.time()
    print('total time: ', end_time - start_time)

if __name__ == '__main__':
    main()
