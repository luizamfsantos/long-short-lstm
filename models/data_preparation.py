import torch
import lightning as L
from pathlib import Path
from torch.utils.data import (
    Dataset,
    DataLoader)

# TODO: where can transformations be applied to the data?

class TimeSeriesDataSingleBatch(Dataset):
    def __init__(
        self,
        sequence_length: int,
        input_tensor_path: str,
        target_tensor_path: str):
        """ Single batch of time series data.
        Args:
        sequence_length: number of time steps in the data
        input_tensor_path: path to input tensor
        target_tensor_path: path to target tensor

        Attributes:
        sequence_length: number of time steps in the data
        input_tensor: tensor with shape (num_tickers, timestamps, features)
        target_tensor: tensor with shape (num_tickers, timestamps, 1)
        """
        self.sequence_length = sequence_length 
        self.input_tensor = torch.load(input_tensor_path) 
        self.target_tensor = torch.load(target_tensor_path) 

    def __len__(self):
        """ Returns the number of sequences in the dataset. 
        Used by the DataLoader to split the data into batches. """
        return self.input_tensor.shape[1] - self.sequence_length

    def __getitem__(self, idx):
        """ Returns a sequence of input and target tensors.
        input_tensor: tensor with shape (num_tickers, sequence_length, features)
        target_tensor: tensor with shape (num_tickers, 1)
        """
        return self.input_tensor[:, idx:idx+self.sequence_length, :], \
            self.target_tensor[:, idx+self.sequence_length, :]


class TimeSeriesDataModule(L.LightningDataModule):
    def __init__(
        self, 
        data_dir: str = 'data/processed',
        batch_size: int = 1,
        sequence_length: int = 4
        ):
        """ Time series data module.
        Args:
        data_dir: directory containing input and target tensors
            train tensors should be named input_train.pt and target_train.pt
            test tensors should be named input_test.pt and target_test.pt
        batch_size: number of sequences until the model is updated
        sequence_length: number of timestamps the model will look at to make a prediction
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size 
        self.sequence_length = sequence_length
        assert self._check_data_exists(), 'Data does not exist. ' \
                                          'Run data_preparation.py to generate data.'

    def _check_data_exists(self):
        """ Check if the input and target tensors exist. """
        input_train_path = f'{self.data_dir}/input_train.pt'
        target_train_path = f'{self.data_dir}/target_train.pt'
        input_test_path = f'{self.data_dir}/input_test.pt'
        target_test_path = f'{self.data_dir}/target_test.pt'
        return all(
            map(
                lambda x: x.exists(),
                [
                    Path(input_train_path),
                    Path(target_train_path),
                    Path(input_test_path),
                    Path(target_test_path)
                ]
            )
        )

    def setup(self, stage=None):
        """ Load the input and target tensors. 
        Stage is used by Lightning to determine if 
        the data is for training or testing.
        Called by Lightning when the data module is 
        initialized.
        """
        self.train_data = TimeSeriesDataSingleBatch(
            sequence_length=self.sequence_length,
            input_tensor_path=f'{self.data_dir}/input_train.pt',
            target_tensor_path=f'{self.data_dir}/target_train.pt'
        )
        self.test_data = TimeSeriesDataSingleBatch(
            sequence_length=self.sequence_length,
            input_tensor_path=f'{self.data_dir}/input_test.pt',
            target_tensor_path=f'{self.data_dir}/target_test.pt'
        )
    
    def train_dataloader(self):
        """ Returns a DataLoader for the training data. """
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          shuffle=False)

    def test_dataloader(self):
        """ Returns a DataLoader for the test data. """
        return DataLoader(self.test_data,
                          batch_size=self.batch_size,
                          shuffle=False)

# example usage
# data_module = TimeSeriesDataModule()