import os.path as osp
import torch
from zipfile import ZipFile

import pandas as pd
from torch_geometric.data import Dataset, download_url


class test_NetlistGraphDataset(Dataset):
    def __init__(self, root, csv_DIR,filePath, transform=None, pre_transform=None):
        self.filePath = osp.join(csv_DIR, filePath)
        super(test_NetlistGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        fileDF = pd.read_csv(self.filePath)
        #print(fileDF)
        return fileDF['fileName'].tolist()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        filePathArchive = osp.join(self.processed_dir, self.processed_file_names[idx])
        filePathName = osp.basename(osp.splitext(filePathArchive)[0])
        with ZipFile(filePathArchive) as myzip:
            with myzip.open(filePathName) as myfile:
                data = torch.load(myfile)
        return data

