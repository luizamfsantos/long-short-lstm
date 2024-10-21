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
    def __init__(self,
                 seq_len: int = 4,
                 tensor_path: str = 'data/processed/tensor_batches',
                 target_path: str = 'data/processed/target_batches',
                 metadata_path: str = 'data/processed/metadata.pt'):
        """
        The tensors have the shape [number_tickers, number_timestamps, feature_vector]
        Target variable have the shape [number_tickers, number_timestamps, 1]
        """
        self.seq_len = seq_len
        self.tensor_path = tensor_path
        self.target_path = target_path
        self.metadata_path = metadata_path
        self.generator = load_tensors(tensor_path, target_path, metadata_path)
        self.current_X, self.current_y, _ = next(self.generator)  # Load the first batch
        self.batch_length = self.current_X.shape[1] # Number of timestamps in the current batch
        self.total_length = 0
        self.batch_start_idx = 0


    def __len__(self):
        return self.total_length # Total number of sequences in the dataset

    def _load_next_batch(self):
        """ Load the next batch from the generator """
        try:
            self.current_X, self.current_y, _ = next(self.generator)  # Load the next batch
            self.batch_length = self.current_X.shape[1]
            self.total_length += self.batch_length
            self.batch_start_idx = self.total_length - self.batch_length
        except StopIteration:
            raise IndexError("No more data available in the generator.")

    def __getitem__(self, idx):
        """
        Fetch a sequence of length `seq_len` at global index `idx`.
        Handles batch transitions seamlessly.
        """
        # Check if the index is within the current batch
        while idx >= self.batch_start_idx + self.batch_length:
            self._load_next_batch()
        
        local_idx = idx - self.batch_start_idx # Index within the current batch
        # Check if the sequence spans two batches
        if local_idx + self.seq_len > self.batch_length:
            first_part_len = self.batch_length - local_idx
            second_part_len = self.seq_len - first_part_len
            X_first = self.current_X[:, local_idx:local_idx:, :]
            y_first = self.current_y[:, local_idx:local_idx:, :]

            self._load_next_batch()
            X_second = self.current_X[:, :second_part_len, :]
            y_second = self.current_y[:, :second_part_len, :]
            X = torch.cat([X_first, X_second], dim=1) # Concatenate the two parts
            y = torch.cat([y_first, y_second], dim=1)
        else:
            X = self.current_X[:, local_idx:local_idx+self.seq_len, :]
            y = self.current_y[:, local_idx:local_idx+self.seq_len, :]
        return X, y

# example usage
# train_dataset = TimeseriesDataset(seq_len = 5)
# train_loader = DataLoader(train_dataset, shuffle = False)