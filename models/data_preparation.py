import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.optim import Adam
import lightning as L
from torch.utils.data import (
    Dataset,
    DataLoader)
import pyarrow.dataset as ds


class TimeSeriesData(Dataset):
    def __init__(self, folder_path: str):
        self.dataset = ds.dataset(folder_path, format='parquet')

    def __len__(self):
        if not (hasattr(self, 'total_rows')):
            self.total_rows = sum(file.count_rows()
                                  for file in self.dataset.get_fragments())
        return self.total_rows

    def __getitem__(self):
        pass
