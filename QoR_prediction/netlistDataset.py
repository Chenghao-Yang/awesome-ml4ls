import os.path as osp
import torch
from zipfile import ZipFile

import pandas as pd
from torch_geometric.data import Dataset, download_url


class NetlistGraphDataset(Dataset):
    def __init__(self, root, csv_DIR,filePath, transform=None, pre_transform=None):
        self.filePath = osp.join(csv_DIR, filePath)
        self.root_path = root
        super(NetlistGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        fileDF = pd.read_csv(self.filePath)
        #print(fileDF)
        return fileDF['fileName'].tolist()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        fileNmae = self.processed_file_names[idx]
        circuit_name = (osp.splitext(fileNmae)[0]).split('_syn')[0]



        filePathArchive = osp.join(self.root_path, circuit_name,fileNmae)
        #print(filePathArchive)
        #filePathName = osp.basename(osp.splitext(filePathArchive)[0])
        """
        with ZipFile(filePathArchive) as myzip:
            with myzip.open(filePathName) as myfile:
                data = torch.load(myfile)
        """
        data = torch.load(filePathArchive)
        return data

