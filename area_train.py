import os

import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import *
from utils import *
from netlistDataset import *
from test_netlistDataset import *

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import random_split
import os.path as osp
import pickle
import sys
import time
import csv

datasetDict = {
    'set1' : ["train_data_mixmatch_v1.csv","valid_data_mixmatch_v1.csv", "test_data_mixmatch_v1.csv"],
    'set2' : ["train_small.csv","valid_small.csv", "test_small.csv"],
    'set3' : ["train_data.csv","valid_data.csv", "test_data.csv"]
}

DUMP_DIR = None
criterion = torch.nn.MSELoss()

def plotChart(x, y_valid,y_train, xlabel,ylabel,leg_label_valid,leg_label_train):
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(x,y_valid, label=leg_label_valid)
    plt.plot(x, y_train, label=leg_label_train)
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.xlabel(xlabel, weight='bold')
    plt.ylabel(ylabel, weight='bold')
    #plt.title(title,weight='bold')
    plt.savefig(osp.join(DUMP_DIR,'train_curve.svg'), format='svg', bbox_inches='tight', dpi=300)

def train(model,device,dataloader,optimizer):
    epochLoss = 0
    model.train()
    n = 0
    for step, batch in enumerate(tqdm(dataloader, desc="Iteration",file=sys.stdout)):
        n += 1
        batch = batch.to(device)
        lbl = batch.target_area.reshape(-1, 1)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred,lbl)
        loss.backward()
        optimizer.step()
        epochLoss += loss.detach().item()
    epochLoss = epochLoss / n
    return epochLoss


def evaluate(model, device, dataloader):
    model.eval()
    validLoss = 0
    n = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration",file=sys.stdout)):
            n += 1
            batch = batch.to(device)
            pred = model(batch)
            lbl = batch.target_area.reshape(-1, 1)
            mseVal = mse(pred, lbl)
            validLoss += mseVal
        validLoss = validLoss / n
    return validLoss


def test(model, device, dataloader):
    model.eval()
    testLoss = 0
    total_mae = 0
    n = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration",file=sys.stdout)):
            n += 1
            batch = batch.to(device)
            pred = model(batch)
            lbl = batch.target_area.reshape(-1, 1)
            accVal = accuracy(pred, lbl)
            mae_Val = mae(pred, lbl)
            testLoss += accVal
            total_mae += mae_Val
        testLoss = testLoss / n
        total_mae = total_mae / n
        rate = np.round(testLoss, 6)
        #print('rate.shape---->', rate)
        # accuracy_rate = "%.2f%%" % (rate * 100)
        accuracy_rate = format(rate, '.3%')
    return accuracy_rate, total_mae


def evaluate_plot(model, device, dataloader, meanVarNodesDict):
    model.eval()
    totalMSE = 0
    n = 0
    batchData = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration", file=sys.stdout)):
            n += 1
            batch = batch.to(device)
            desName = batch.desName
            pred = model(batch)
            #print('pred---->', pred)
            lbl = batch.target_area.reshape(-1, 1)
            pred_ununnormalized = pred.reshape(-1, 1)
            lbl_unnormalized = lbl
            #print('pred_ununnormalized---->', pred_ununnormalized.type())
            #print('lbl_unnormalized---->', lbl_unnormalized.type())
            predArray = pred.view(-1, 1).detach().cpu().numpy() #* meanVarNodesDict['usb_phy'][1] + meanVarNodesDict['usb_phy'][0]
            actualArray = lbl.view(-1, 1).detach().cpu().numpy() #* meanVarNodesDict['usb_phy'][1] + meanVarNodesDict['usb_phy'][0]
            #print(meanVarNodesDict)

            for i in range(len(pred)):
                de_name = desName[i][0]
                predArray[i] = predArray[i] * meanVarNodesDict[de_name][1] + meanVarNodesDict[de_name][0]
                actualArray[i] = actualArray[i] * meanVarNodesDict[de_name][1] + meanVarNodesDict[de_name][0]

            predArray  = np.round(predArray, 0)
            actualArray = np.round(actualArray, 0)
            #print('predArray--->', predArray)
            #print('actualArray---->', actualArray)

            batchData.append([predArray, actualArray, desName])
            #print('batchData---->', batchData)
            mseVal = mse(pred, lbl)
            totalMSE += mseVal
        totalMSE = totalMSE / n
    return totalMSE, batchData



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on Synthesis Task Pytorch Geometric')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lp', type=int, default=1,
                        help='Learning problem (QoR prediction: 1,Classification: 2)')
    parser.add_argument('--epochs', type=int, default=80,
                        help='number of epochs to train (default: 80)')
    parser.add_argument('--dataset', type=str, default="set1",
                        help='Split strategy (set1/set2/set3 default: set1)')

    args = parser.parse_args()

    datasetChoice = 'set3'
    #RUN_DIR = args.rundir

    # Hyperparameters
    #batchSize = args.batch_size #64
    batchSize = 128
    num_epochs = 2
    learning_rate = 0.001
    """
    num_epochs = args.epoch #80
    learning_rate = args.lr #0.001
    """
    #learningProblem = args.lp
    targetLbl = 'area'
    nodeEmbeddingDim = 31
    synthEncodingDim = 4

    IS_STATS_AVAILABLE = True
    train_ROOT_DIR = "/home/yangch/TODAES/data_set/ptdata/train/" #'/scratch/abc586/OPENABC_DATASET'
    valid_ROOT_DIR = "/home/yangch/TODAES/data_set/ptdata/valid/"
    test_ROOT_DIR = "/home/yangch/TODAES/data_set/test/"
    csv_DIR = "/home/yangch/TODAES/data_set/"
    global DUMP_DIR
    DUMP_DIR = "/home/yangch/TODAES/RUN_DIR/area/" #osp.join('/scratch/abc586/OpenABC-dataset/SynthV9_AND',RUN_DIR)

    if not osp.exists(DUMP_DIR):
        os.mkdir(DUMP_DIR)


    # Load train and test datasets
    trainDS = NetlistGraphDataset(root=osp.join(train_ROOT_DIR),csv_DIR=osp.join(csv_DIR),filePath=datasetDict[datasetChoice][0])
    validDS = NetlistGraphDataset(root=osp.join(valid_ROOT_DIR),csv_DIR=osp.join(csv_DIR),filePath=datasetDict[datasetChoice][1])
    testDS = test_NetlistGraphDataset(root=osp.join(test_ROOT_DIR),csv_DIR=osp.join(csv_DIR),filePath=datasetDict[datasetChoice][2])
    #print(trainDS)

    if IS_STATS_AVAILABLE:
        with open(osp.join(csv_DIR,'synthesisStatistics.pickle'),'rb') as f:
            numGatesAndLPStats = pickle.load(f)
    else:
        print("\nNo pickle file found for number of gates")
        exit(0)

    meanVarNodesDict = computeMeanAndVarianceOfTargets(numGatesAndLPStats, targetVar=targetLbl)
    """
    trainDS.transform = transforms.Compose([lambda data: train_addNormalizedTargets(data,numGatesAndLPStats,meanVarNodesDict,targetVar=targetLbl)])
    validDS.transform = transforms.Compose([lambda data: valid_addNormalizedTargets(data,numGatesAndLPStats, meanVarNodesDict,targetVar=targetLbl)])
    testDS.transform = transforms.Compose([lambda data: test_addNormalizedTargets(data,numGatesAndLPStats,meanVarNodesDict,targetVar=targetLbl)])
    """
    num_classes = 1


    # Define the model
    synthFlowEncodingDim = trainDS[0].synVec.size()[0]*synthEncodingDim
    node_encoder = NodeEncoder(emb_dim=nodeEmbeddingDim)
    synthesis_encoder = SynthFlowEncoder(emb_dim=synthEncodingDim)

    model = SynthNet(node_encoder=node_encoder,synth_encoder=synthesis_encoder,n_classes=num_classes,synth_input_dim=synthFlowEncodingDim,node_input_dim=nodeEmbeddingDim)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',verbose=True)
    device = getDevice()
    model = model.to(device)

    # Split the training data into training and validation dataset
    #training_validation_samples = [int(0.8*len(trainDS)),len(trainDS)-int(0.8*len(trainDS))]
    #train_DS,valid_DS = random_split(trainDS,training_validation_samples)


    # Initialize the dataloaders
    train_dl = DataLoader(trainDS,shuffle=True,batch_size=batchSize,pin_memory=True,num_workers=16,persistent_workers=True)
    valid_dl = DataLoader(validDS,shuffle=False,batch_size=batchSize,pin_memory=True,num_workers=16,persistent_workers=True)
    test_dl = DataLoader(testDS,shuffle=False,batch_size=batchSize,pin_memory=True,num_workers=16,persistent_workers=True)


    # Monitor the loss parameters
    valid_curve = []
    train_loss = []


    for ep in range(1, num_epochs + 1):
        print("\nEpoch [{}/{}]".format(ep, num_epochs))
        print("\nTraining..")
        trainLoss = train(model, device, train_dl, optimizer)

        print("\nEvaluation..")
        validLoss = evaluate(model, device, valid_dl)

        print({'Train loss': trainLoss,'Validation loss': validLoss})
        valid_curve.append(validLoss)
        train_loss.append(trainLoss)
        torch.save(model.state_dict(), osp.join(DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(ep, validLoss)))
        scheduler.step(validLoss)

    best_val_epoch = np.argmax(np.array(valid_curve))

    csvFile2 = open(os.path.join(DUMP_DIR, "training_data" + '.csv'), 'a', newline='')  # 设置newline，否则两行之间会空一行
    writer = csv.writer(csvFile2)
    writer.writerow(['train', 'valid'])
    for i in range(len(valid_curve)):
        train_date = train_loss[i]
        valid_data = valid_curve[i]
        writer.writerow([train_date, valid_data])

    testAcc = evaluate(model, device, test_dl)
    print("\nTest loss.. :" + str(testAcc))
    testAccuracy_rate, test_MAE = test(model, device, test_dl)
    print("\nTest Accuracy:" + str(testAccuracy_rate) + "\nTest MAE:" + str(test_MAE))

    writer.writerow(['Test loss', 'MAPE', 'MAE'])
    writer.writerow([testAcc, testAccuracy_rate, test_MAE])
    csvFile2.close()


    # Save training data for future plots
    with open(osp.join(DUMP_DIR,'valid_curve.pkl'),'wb') as f:
        pickle.dump(valid_curve,f)

    with open(osp.join(DUMP_DIR,'train_loss.pkl'),'wb') as f:
        pickle.dump(train_loss,f)

    ##### EVALUATION ######
    plotChart([i+1 for i in range(len(valid_curve))],valid_curve,train_loss, "Epochs","Loss","Validation loss","Training loss")
    #plotChart([i+1 for i in range(len(train_loss))],train_loss,"# Epochs","Loss","train_loss","Training loss")

    # Evaluate on train data
    trainMSE,trainBatchData = evaluate_plot(model, device, train_dl, meanVarNodesDict)
    NUM_BATCHES_TRAIN = len(train_dl)
    doScatterPlot(NUM_BATCHES_TRAIN,batchSize,trainBatchData,DUMP_DIR,"train")

    # Evaluate on validation data
    validMSE,validBatchData = evaluate_plot(model, device, valid_dl, meanVarNodesDict)
    NUM_BATCHES_VALID = len(valid_dl)
    doScatterPlot(NUM_BATCHES_VALID,batchSize,validBatchData,DUMP_DIR,"valid")

    # Evaluate on test data
    testMSE,testBatchData = evaluate_plot(model, device, test_dl, meanVarNodesDict)
    NUM_BATCHES_TEST = len(test_dl)
    doScatterPlot(NUM_BATCHES_TEST,batchSize,testBatchData,DUMP_DIR,"test")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time - start_time, '秒')