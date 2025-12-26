import os
import pandas as pd
import numpy as np
from torchvision.io import decode_image
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # get features, label using index:
        item_features = self.features.iloc[idx]
        item_labels = self.labels.iloc[idx]
        # move to ndarray:
        item_features = item_features.to_numpy().astype(np.float32)
        item_labels = item_labels.to_numpy().astype(np.float32)
        return item_features, item_labels


