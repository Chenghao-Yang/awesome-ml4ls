import torch
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd
import numpy as np

def getMeanAndVariance(targetList):
    return np.mean(np.array(targetList)),np.std(np.array(targetList))

def computeMeanAndVarianceOfTargets(targetStatsDict,targetVar='nodes'):
    meanAndVarTargetDict = {}
    for des in targetStatsDict.keys():
        #numNodes,_,_,areaVar,delayVar = targetStatsDict[des]
        areaVar, delayVar = targetStatsDict[des]
        if targetVar == 'delay':
            meanTarget,varTarget = getMeanAndVariance(delayVar)
        elif targetVar == 'area':
            meanTarget,varTarget = getMeanAndVariance(areaVar)
        else:
            meanTarget,varTarget = getMeanAndVariance(numNodes)
        meanAndVarTargetDict[des] = [meanTarget,varTarget]
    return meanAndVarTargetDict

def train_addNormalizedTargets(data,targetStatsDict,meanVarDataDict,targetVar='nodes'):
    sid = data.synID[0]
    desName = data.desName[0]
    if sid < 60000:
        sid = sid
    if 199999 < sid < 215000:
        sid = sid - 140000

    if targetVar == 'delay':
        targetIdentifier = 1 # Column number of target 'Delay' in synthesisStatistics.pickle entries
        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    elif targetVar == 'area':
        targetIdentifier = 0 # Column number of target 'Area' in synthesisStatistics.pickle entries
        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    else:
        targetIdentifier = 0 # Column number of target 'Nodes' in synthesisStatistics.pickle entries
        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    return data

def valid_addNormalizedTargets(data,targetStatsDict,meanVarDataDict,targetVar='nodes'):
    sid = data.synID[0]
    desName = data.desName[0]

    if 59999 < sid < 70000:
        sid = sid - 60000
    if 214999 < sid < 217500:
        sid = sid - 205000

    if targetVar == 'delay':
        targetIdentifier = 1 # Column number of target 'Delay' in synthesisStatistics.pickle entries
        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    elif targetVar == 'area':
        targetIdentifier = 0 # Column number of target 'Area' in synthesisStatistics.pickle entries
        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    else:
        targetIdentifier = 0 # Column number of target 'Nodes' in synthesisStatistics.pickle entries
        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    return data

def test_addNormalizedTargets(data,targetStatsDict,meanVarDataDict,targetVar='nodes'):
    sid = data.synID[0]
    desName = data.desName[0]

    if 69999 < sid < 200000:
        sid = sid - 70000
    if 217499 < sid < 250000:
        sid = sid - 87500

    if targetVar == 'delay':
        targetIdentifier = 1 # Column number of target 'Delay' in synthesisStatistics.pickle entries
        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    elif targetVar == 'area':
        targetIdentifier = 0 # Column number of target 'Area' in synthesisStatistics.pickle entries
        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    else:
        targetIdentifier = 0 # Column number of target 'Nodes' in synthesisStatistics.pickle entries
        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    return data


def addAbsoluteTargets(data,targetStatsDict,targetVar='nodes'):
    sid = data.synID[0]
    desName = data.desName[0]
    numNodes,_,_,areaVar,delayVar = targetStatsDict[desName]
    if targetVar == 'delay':
        data.target = torch.tensor([delayVar[sid]],dtype=torch.float32)
    elif targetVar == 'area':
        data.target = torch.tensor([areaVar[sid]],dtype=torch.float32)
    else:
        data.target = torch.tensor([numNodes[sid]],dtype=torch.float32)
    return data

# Torch.std_mean returns tuple with std first and mean second term
def mapMeanChangeToTensor(data,areaStatsDict,delayStatsDict):
    area = data.area
    delay = data.delay
    data.area = (area - areaStatsDict[data.desName[0]][1]) / areaStatsDict[data.desName[0]][0]
    data.delay = (delay - delayStatsDict[data.desName[0]][1]) / delayStatsDict[data.desName[0]][0]
    assert(data.area > -10 and data.area < 10)
    return data

# Element 0 is area and 1 is delay
def getMeanAreaAndDelay(trainDS,testDS):
    desNamesTrain = set(elem.desName[0] for elem in trainDS)
    desNamesTest = set(elem.desName[0] for elem in testDS)
    desNameTotal = desNamesTrain.union(desNamesTest)
    desStatsArea = {}
    desStatsDelay = {}
    delayStats = {}
    areaStats = {}
    for des in desNameTotal:
        desStatsArea[des] = []
        desStatsDelay[des] = []
    for elem in trainDS:
        desStatsArea[elem.desName[0]].append(elem.area)
        desStatsDelay[elem.desName[0]].append(elem.delay)
    for elem in testDS:
        desStatsArea[elem.desName[0]].append(elem.area)
        desStatsDelay[elem.desName[0]].append(elem.delay)
    for des in desNameTotal:
        areaStats[des] = torch.std_mean(torch.tensor(desStatsArea[des]))
        delayStats[des] = torch.std_mean(torch.tensor(desStatsDelay[des]))
    return areaStats,delayStats


def getMinMaxTargetVal(dataSet):
    desMinMaxAreaVal = {}
    desMinMaxDelayVal = {}
    desNames = [elem.desName[0] for elem in dataSet]
    for des in desNames:
        desMinMaxAreaVal[des] = [None,None]
        desMinMaxDelayVal[des] = [None,None]
    for ditem in dataSet[1:]:
        des = ditem.desName[0]
        area = ditem.area
        delay = ditem.delay
        # Area computation
        desMinMaxAreaVal[des][0] = area if (area > desMinMaxAreaVal[des][0] or desMinMaxAreaVal[des][0] == None) else desMinMaxAreaVal[des][0]
        desMinMaxAreaVal[des][1] = area if (area < desMinMaxAreaVal[des][1] or desMinMaxAreaVal[des][1] == None) else desMinMaxAreaVal[des][1]
        # Delay computation
        desMinMaxDelayVal[des][0] = delay if (delay > desMinMaxDelayVal[des][0] or desMinMaxDelayVal[des][1] == None) else desMinMaxDelayVal[des][0]
        desMinMaxDelayVal[des][1] = delay if (delay < desMinMaxDelayVal[des][1] or desMinMaxDelayVal[des][1] == None) else desMinMaxDelayVal[des][1]
    return desMinMaxAreaVal,desMinMaxDelayVal

def checkUnseenDesInTest(areaDict,testDS):
    unseenDesigns = set(elem.desName[0] for elem in testDS if not elem.desName[0] in areaDict.keys())
    if len(unseenDesigns) > 0:
        desMinMaxAreaVal = {}
        desMinMaxDelayVal = {}
        for des in unseenDesigns:
            desMinMaxAreaVal[des] = [0, -1]
            desMinMaxDelayVal[des] = [0,-1]
        for ditem in testDS:
            des = ditem.desName[0]
            area = ditem.area
            delay = ditem.delay
            if( not des in unseenDesigns):
                pass
            # Area computation
            desMinMaxAreaVal[des][0] = area if area > desMinMaxAreaVal[des][0] else desMinMaxAreaVal[des][0]
            desMinMaxAreaVal[des][1] = area if (area < desMinMaxAreaVal[des][1] or area == -1) else desMinMaxAreaVal[des][1]
            # Delay computation
            desMinMaxDelayVal[des][0] = delay if delay > desMinMaxDelayVal[des][0] else desMinMaxDelayVal[des][0]
            desMinMaxDelayVal[des][1] = delay if (delay < desMinMaxDelayVal[des][1] or delay == -1) else desMinMaxDelayVal[des][1]
        return desMinMaxAreaVal, desMinMaxDelayVal
    else:
        return None,None


def getDevice():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def desName_to_idx(aigData):
    desNames = [elem.desName[0] for elem in aigData]
    desNameIdxDict = {}
    idxDesNameDict = {}
    i=0
    for des in desNames:
        if not des in desNameIdxDict.keys():
            desNameIdxDict[des] = i
            idxDesNameDict[i] = des
            i+=1
    return desNameIdxDict,idxDesNameDict

def mapNameToLabel(data,desNameIdxDict):
    labelName = data.desName[0]
    data.desLabel = torch.tensor([desNameIdxDict[labelName]])
    return data

def mapAttributesToTensor(data,areaDict,delayDict):
    area = data.area
    delay = data.delay
    minMaxArea = areaDict[data.desName[0]]
    minMaxDelay = delayDict[data.desName[0]]
    data.area = (area - minMaxArea[1])/(minMaxArea[0] - minMaxArea[1])
    data.delay = (delay - minMaxDelay[1]) / (minMaxDelay[0] - minMaxDelay[1])
    return data


def mse(y_pred,y_true):
    return mean_squared_error(y_true.view(-1,1).detach().cpu().numpy(),y_pred.view(-1,1).detach().cpu().numpy())

def mae(y_pred,y_true):
    return mean_absolute_error(y_true.view(-1,1).detach().cpu().numpy(),y_pred.view(-1,1).detach().cpu().numpy())

def accuracy(y_pred, y_true):
    y_true = y_true.view(-1,1).detach().cpu().numpy()
    y_pred = y_pred.view(-1,1).detach().cpu().numpy()
    erro = y_true - y_pred
    lenth = len(y_true)
    #rate = abs((y_true - y_pred)) / np.clip(np.abs(y_true), 1e-6, None)
    rate = abs((y_true - y_pred) / y_true)
    #rate = sum(rate) / lenth
    rate = np.mean(rate)
    return rate

def mape(y_true,y_pred):
    y_true = y_true.view(-1,1).detach().cpu().numpy()
    y_pred = y_pred.view(-1,1).detach().cpu().numpy()
    return np.mean(np.abs((y_pred-y_true)/y_true))

def doScatterPlot(batchLen,batchSize,batchData,dumpDir,trainMode):
    predList = []
    actualList = []
    designList = []
    for i in range(batchLen):
        numElemsInBatch = len(batchData[i][0])
        for batchID in range(numElemsInBatch):
            predList.append(batchData[i][0][batchID][0])
            actualList.append(batchData[i][1][batchID][0])
            designList.append(batchData[i][2][batchID][0])

    scatterPlotDF = pd.DataFrame({'designs': designList,
                                  'prediction': predList,
                                  'actual': actualList})

    uniqueDesignList = scatterPlotDF.designs.unique()
    
    plt.style.use('seaborn')
    
    for d in uniqueDesignList:
        designDF = scatterPlotDF[scatterPlotDF.designs == d]
        designDF.plot.scatter(x='actual', y='prediction', c='DarkBlue')
        plt.title(d)
        fileName = osp.join(dumpDir,"scatterPlot_"+trainMode+"_"+d+".svg")
        #else:
        #    fileName = osp.join(dumpDir,"scatterPlot_test_"+d+".png")
        plt.savefig(fileName,format='svg',bbox_inches='tight')


def getTopKSimilarityPercentage(list1,list2,topkpercent):
    listLen = len(list1)
    topKIndexSimilarity = int(topkpercent*listLen)
    Set1 = set(list1[:topKIndexSimilarity])
    Set2 = set(list2[:topKIndexSimilarity])
    numSimilarScripts = len(Set1.intersection(Set2))
    return (numSimilarScripts/listLen)


def doScatterAndTopKRanking(batchLen,batchSize,batchData,dumpDir,trainMode):
    predList = []
    actualList = []
    designList = []
    synthesisID = []
    for i in range(batchLen):
        numElemsInBatch = len(batchData[i][0])
        for batchID in range(numElemsInBatch):
            predList.append(batchData[i][0][batchID][0])
            actualList.append(batchData[i][1][batchID][0])
            designList.append(batchData[i][2][batchID][0])
            synthesisID.append(batchData[i][3][batchID][0])

    scatterPlotDF = pd.DataFrame({'designs': designList,
                                  'synID': synthesisID,
                                  'prediction': predList,
                                  'actual': actualList})

    uniqueDesignList = scatterPlotDF.designs.unique()

    print("\nDataset type: "+trainMode)
    for d in uniqueDesignList:
        designDF = scatterPlotDF[scatterPlotDF.designs == d]
        designDF.plot.scatter(x='actual', y='prediction', c='DarkBlue')
        plt.title(d,weight='bold',fontsize=25)
        plt.xlabel('Actual', weight='bold', fontsize=25)
        plt.ylabel('Predicted', weight='bold', fontsize=25)
        fileName = osp.join(dumpDir,"scatterPlot_"+trainMode+"_"+d+".png")
        plt.savefig(fileName,fmt='png',bbox_inches='tight')
        desDF1 = designDF.sort_values(by=['actual'])
        desDF2 = designDF.sort_values(by=['prediction'])
        desDF1_synID = desDF1.synID.to_list()
        desDF2_synID = desDF2.synID.to_list()
        top5percentScore = getTopKSimilarityPercentage(desDF1_synID,desDF2_synID,0.05)
        top10percentScore = getTopKSimilarityPercentage(desDF1_synID,desDF2_synID,0.10)
        top20percentScore = getTopKSimilarityPercentage(desDF1_synID, desDF2_synID,0.20)
        print("\n"+d+" top 5:"+str(top5percentScore)+" top 10:"+str(top10percentScore)+" top 20:"+str(top20percentScore))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count