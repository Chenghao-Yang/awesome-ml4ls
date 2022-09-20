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


IS_STATS_AVAILABLE = True

if IS_STATS_AVAILABLE:
    with open(osp.join("/home/yangch/TODAES/data_set/", 'synthesisStatistics.pickle'), 'rb') as f:
        targetStatsDict = pickle.load(f)
else:
    print("\nNo pickle file found for number of gates")
    exit(0)


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

meanVarTargetDict = computeMeanAndVarianceOfTargets(targetStatsDict)


def getAllStatistics(adpPath):
    desDict = {}
    adpCsvFiles = glob.glob(osp.join(adpPath,"*.csv"))
    #finalAigCsvFiles = glob.glob(osp.join(finalAigPath,"*.csv"))
    designFileDicts = {}
    # for csvF in finalAigCsvFiles:
    #     desName = osp.basename(csvF).split("processed_")[1].split(".csv")[0]
    #     designFileDicts[desName] = [csvF]
    for csvF in adpCsvFiles:
        desName = osp.basename(csvF).split("adp_")[1].split(".csv")[0]
        designFileDicts[desName] = [csvF]

    for des in designFileDicts.keys():
        # if (len(designFileDicts[des]) < 2):
        #     continue # Should have both the statistics available
        # desDF_fAIG = pd.read_csv(designFileDicts[des][0])
        # desDF_fAIG = desDF_fAIG.sort_values(['sid'], ascending=True)
        # ANDgates = desDF_fAIG["AND"].tolist()
        # NOTgates = desDF_fAIG["NOT"].tolist()
        # lpLen = desDF_fAIG["LP"].tolist()

        desDF_adp = pd.read_csv(designFileDicts[des][0])
        desDF_adp = desDF_adp.sort_values(['sid'], ascending=True)
        area = desDF_adp["area"]#.tolist()
        delay = desDF_adp["delay"]#.tolist()

        #delay_targetIdentifier = 1  # Column number of target 'Delay' in synthesisStatistics.pickle entries
        normTarget_delay = (area - meanVarTargetDict[des][2]) / meanVarTargetDict[des][3]

        #area_targetIdentifier = 0  # Column number of target 'Area' in synthesisStatistics.pickle entries
        normTarget_area = (delay - meanVarTargetDict[des][0]) / meanVarTargetDict[des][1]

        normTarget_delay = normTarget_delay.tolist()
        normTarget_area = normTarget_area.tolist()


        desDict[des] = [normTarget_area,normTarget_delay]

    with open(osp.join("/home/yangch/TODAES/data_set/statistics/","nor_synthesisStatistics.pickle"),'wb') as f:
        pickle.dump(desDict,f)



def main():
    start_time = time.time()

    adpPath = "/home/yangch/TODAES/data_set/statistics/adp/"
    getAllStatistics(adpPath)

    end_time = time.time()
    print('total time: ', end_time - start_time)


if __name__=='__main__':
    main()