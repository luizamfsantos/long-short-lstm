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

# TODO: where can transformations be applied to the data?

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
        self.total_length = self.batch_length # Total number of timestamps loaded
        self.batch_start_idx = 0

    def __len__(self):
        """ This is used to calculate the total length of the dataset. 
        It is the total number of timestamps minus the sequence length.
        It is called by the DataLoader to determine the number of batches."""
        mock_generator = load_tensors(self.tensor_path, self.target_path, self.metadata_path)
        total_length = 0
        for batch_input, batch_output, _ in mock_generator:
            total_length += batch_input.shape[1]
        return total_length - self.seq_len

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
        # Check if the index is within the batches that have been loaded
        while idx >= self.total_length: # this will happen when the the idx is exactly the same as the total_length + 1
            self._load_next_batch()
        
        local_idx = idx - self.batch_start_idx # Index within the current batch
        # Check if the sequence spans two batches
        if local_idx + self.seq_len > self.batch_length:
            # Split into 2 parts
            first_part_len = self.batch_length - local_idx
            second_part_len = self.seq_len - first_part_len
            # First part of the sequence
            X_first = self.current_X[:, local_idx:, :]
            y_first = self.current_y[:, local_idx:, :]
            # Second part of the sequence
            self._load_next_batch()
            X_second = self.current_X[:, :second_part_len, :]
            y_second = self.current_y[:, :second_part_len, :]
            # Concatenate the two parts
            X = torch.cat([X_first, X_second], dim=1) 
            y = torch.cat([y_first, y_second], dim=1)
        else:
            X = self.current_X[:, local_idx:local_idx+self.seq_len, :]
            y = self.current_y[:, local_idx:local_idx+self.seq_len, :]
        return X, y

class TimeSeriesDataSingleBatch(Dataset):
    def __init__(
        self,
        seq_len: int,
        input_tensor_path: str,
        target_tensor_path: str):
        self.seq_len = seq_len
        self.input_tensor = torch.load(input_tensor_path)
        self.target_tensor = torch.load(target_tensor_path)

    def __len__(self):
        return self.input_tensor.shape[1] - self.seq_len

    def __getitem__(self, idx):
        return self.input_tensor[:, idx:idx+self.seq_len, :], \
               self.target_tensor[:, idx:idx+self.seq_len, :] # TODO: change this to only get the last timestamp of the sequence


class TimeSeriesDataModule(L.LightningDataModule):
    def __init__(
        self, 
        data_dir: str = 'data/processed',
        batch_size: int = 1,
        seq_len: int = 4
        ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size # this is the number of sequences until the model is updated
        self.seq_len = seq_len # this is how many timestamps the model will look at to make a prediction

    def setup(self, stage=None):
        self.data_train = TimeSeriesDataSingleBatch(
            seq_len=self.seq_len,
            input_tensor_path=f'{self.data_dir}/input_train.pt',
            target_tensor_path=f'{self.data_dir}/target_train.pt'
        )
        self.data_test = TimeSeriesDataSingleBatch(
            seq_len=self.seq_len,
            input_tensor_path=f'{self.data_dir}/input_test.pt',
            target_tensor_path=f'{self.data_dir}/target_test.pt'
        )
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False)

# example usage
# train_dataset = TimeseriesDataset(seq_len = 5)
# train_loader = DataLoader(train_dataset, batch_size=1,shuffle=False)