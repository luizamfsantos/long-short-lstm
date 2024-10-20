import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.optim import Adam
import lightning as L
from torch.utils.data import (
    Dataset,
    DataLoader)
import pyarrow.dataset as ds
from ingestion.preprocess import read_data

# Step 1: Sort the dataframe by both date and ticker so that we can easily map values to the tensor.
# Step 2: Create a dictionary of tickers and their corresponding indices.
# Step 3: Iterate through the dataframe and create a tensor where each row corresponds to a ticker and each column represents the features for a specific timestamp.
# Step 4: Create a tensor for the target variable.

class TimeSeriesData(Dataset):
    def __init__(self, folder_path: str, batch_size = 10000):
        """ Batch size is the number of rows to read at a time
        The raw data is organized by data and the columns are the features.
        If there are 200 tickers, and batch_size is 1000, 
        then the first batch will have 1000/200 = 5 rows per ticker.
        """
        self.folder_path = folder_path
        self.data_generator = read_data(folder_path, batch_size=batch_size)
        self.total_rows = len(self)
        self.batch = next(self.data_generator) # get the first batch

    def __len__(self):
        if not (hasattr(self, 'total_rows')):
            self.total_rows = sum(batch.num_rows for batch in self.data_generator)
            self.data_generator = read_data(self.folder_path)  # reset the generator
        return self.total_rows

    def __getitem__(self):
        """ At each iteration """
