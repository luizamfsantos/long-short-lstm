import torch
import lightning as L
from torch.utils.data import (
    Dataset,
    DataLoader)

# TODO: where can transformations be applied to the data?

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
               self.target_tensor[:, idx+self.seq_len, :] 


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
        self.train_data = TimeSeriesDataSingleBatch(
            seq_len=self.seq_len,
            input_tensor_path=f'{self.data_dir}/input_train.pt',
            target_tensor_path=f'{self.data_dir}/target_train.pt'
        )
        self.test_data = TimeSeriesDataSingleBatch(
            seq_len=self.seq_len,
            input_tensor_path=f'{self.data_dir}/input_test.pt',
            target_tensor_path=f'{self.data_dir}/target_test.pt'
        )
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

# example usage
# train_dataset = TimeseriesDataset(seq_len = 5)
# train_loader = DataLoader(train_dataset, batch_size=1,shuffle=False)