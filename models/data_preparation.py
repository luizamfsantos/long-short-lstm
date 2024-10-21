import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.optim import Adam
import lightning as L
from torch.utils.data import (
    Dataset,
    DataLoader)
from ingestion.preprocess import load_tensors

# Step 1: Sort the dataframe by both date and ticker so that we can easily map values to the tensor.
# Step 2: Create a dictionary of tickers and their corresponding indices.
# Step 3: Iterate through the dataframe and create a tensor where each row corresponds to a ticker and each column represents the features for a specific timestamp.
# Step 4: Create a tensor for the target variable.

class TimeSeriesData(Dataset):
    def __init__(self, folder_path: str = 'data/processed'):
        """ The tensors have the shape [number_tickers, number_timestamps, feature_vector]
        Target variable have the shape [number_tickers, number_timestamps, 1]
        data/processed/tensor_batches have the .pt files that contain the tensors
        data/processed/targets have the .pt files that contain the target variables
        """
        self.X, self.y = load_tensors(folder_path) # first tensor batch and target variable
        

    def __len__(self):
        """ return the total number of rows in the tensor batch """
        ...

    def __getitem__(self):
        """ At each iteration return the next row in the tensor """
        ...
