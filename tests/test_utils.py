import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(
        self,
        sequence_length: int,
        num_tickers: int,
        feature_length: int,
        timestamps: int,
    ):
        self.sequence_length = sequence_length
        self.num_tickers = num_tickers
        self.feature_length = feature_length
        self.data = torch.rand(
            (num_tickers, timestamps, feature_length),
            dtype=torch.float32
        )
        self.target = torch.randint(
            2,
            (num_tickers, timestamps, 1)).float()

    def __getitem__(self, index: int) -> torch.Tensor:
        """ Returns:
         in_tensor: shape (num_tickers, sequence_length, feature_length)
        target: shape (num_tickers, 1)"""
        return self.data[:, index:index+self.sequence_length, :], \
            self.target[:, index+self.sequence_length, :]

    def __len__(self) -> int:
        """ Returns: number of samples in the dataset 
        with length equal to sequence_length"""
        return self.data.size(1) - self.sequence_length


class RandomDataModule(L.LightningDataModule):
    def __init__(
        self,
        sequence_length: int,
        num_tickers: int,
        feature_length: int,
        timestamps: int,
        batch_size: int,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_tickers = num_tickers
        self.feature_length = feature_length
        self.batch_size = batch_size
        self.timestamps = timestamps

    def setup(self, stage=None):
        self.train_data = RandomDataset(
            sequence_length=self.sequence_length,
            num_tickers=self.num_tickers,
            feature_length=self.feature_length,
            timestamps=self.timestamps,
        )
        self.test_data = RandomDataset(
            sequence_length=self.sequence_length,
            num_tickers=self.num_tickers,
            feature_length=self.feature_length,
            timestamps=self.timestamps,
        )

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          num_workers=9)

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.batch_size,
                          num_workers=9)