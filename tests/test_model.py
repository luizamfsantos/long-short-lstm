import pytest
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, TensorDataset, Dataset
from models.lstm_model import LSTMModel
import memory_profiler
from training.train_utils import get_config

config = get_config()

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
        return self.data[:, index:index+self.sequence_length, :], \
            self.target[:, index+self.sequence_length, :]

    def __len__(self) -> int:
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

@pytest.fixture
def model_params():
    return {
        'feature_length': 10,
        'hidden_size': 5,
        'sequence_length': 5,
        'batch_size': 1,
        'num_layers': 2,
        'dropout_rate': 0.2,
        'learning_rate': 1e-3,
        'num_epochs': 10,
    }

@pytest.fixture
def data_params():
    return {
        'sequence_length': 5,
        'num_tickers': 3,
        'feature_length': 10,
        'timestamps': 10,
        'batch_size': 1,
    }


def test_model_init(model_params):
    model = LSTMModel(**model_params)

    # check if the model has the correct attributes
    assert isinstance(model, L.LightningModule)
    assert isinstance(model.lstm, nn.LSTM)
    assert isinstance(model.linear, nn.Linear)

    # check lstm parameters
    assert model.lstm.input_size == model_params['feature_length']
    assert model.lstm.hidden_size == model_params['hidden_size']
    assert model.lstm.num_layers == model_params['num_layers']

    # check linear parameters
    assert model.linear.in_features == model_params['hidden_size']
    assert model.linear.out_features == 1

@memory_profiler.profile
def test_forward_pass_shape(model_params, data_params, tmp_path):
    model = LSTMModel(**model_params)
    trainer = Trainer(
        devices='auto',
        accelerator='cpu',
        default_root_dir=tmp_path,
        max_epochs=1,
        log_every_n_steps=1)
    data_module = RandomDataModule(**data_params)
    trainer.fit(model, datamodule=data_module)
    # output = model(in_tensor)

    # batch_size, num_tickers, sequence_length, input_size = in_tensor.size()
    # expected_shape = (batch_size, num_tickers, 1)

    # assert output.size() == expected_shape
    # # check if the output is a probability
    # assert torch.all((output >= 0) & (output <= 1))


# def test_training_step(model_params, sample_batch, tmp_path):
#     model = LSTMModel(**model_params)
#     test_checkpoint_dir = tmp_path / 'test_checkpoint_dir'
#     trainer = Trainer(default_root_dir=test_checkpoint_dir, max_epochs=1, log_every_n_steps=1)
#     train_loader = DataLoader(TensorDataset(
#         *sample_batch), batch_size=1, shuffle=False, num_workers=9, persistent_workers=True)
#     # fit the model to initialize the Trainer logging hooks
#     trainer.fit(model, train_loader)
#     # run training step to test the loss computation
#     # TODO: how does batch_idx work?
#     loss = model.training_step(sample_batch, batch_idx=0)

#     assert isinstance(loss, torch.Tensor)
#     assert loss.requires_grad
#     assert loss.item() >= 0
#     assert not torch.isnan(loss)  # check if the loss is not NaN
#     assert not torch.isinf(loss)  # check if the loss is not infinity


# def test_configure_optimizers(model_params):
#     model = LSTMModel(**model_params)
#     optimizer = model.configure_optimizers()

#     assert isinstance(optimizer, torch.optim.Adam)

#     for param_group in optimizer.param_groups:
#         assert param_group['lr'] == model_params['learning_rate']
#         assert param_group['betas'] == (0.9, 0.999)
#         assert param_group['eps'] == 1e-8
#         assert param_group['weight_decay'] == 0
#         assert param_group['amsgrad'] == False
#         assert param_group['params'] == list(model.parameters())


# @pytest.mark.parametrize('batch_size', [1, 2, 4])
# def test_different_batch_sizes(model_params, batch_size):
#     model = LSTMModel(**model_params)
#     num_tickers = 3
#     sequence_length = model_params['sequence_length']
#     feature_length = model_params['input_size']

#     in_tensor = torch.rand(
#         (batch_size, num_tickers, sequence_length, feature_length))
#     target_tensor = torch.randint(2, (batch_size, num_tickers, 1)).float()
#     output = model(in_tensor)
#     assert output.size() == (batch_size, num_tickers, 1)


# @pytest.mark.parametrize('num_tickers', [1, 3, 5])
# def test_different_ticker_counts(model_params, num_tickers):
#     model = LSTMModel(**model_params)
#     batch_size = 4
#     sequence_length = model_params['sequence_length']
#     feature_length = model_params['input_size']

#     in_tensor = torch.rand(
#         (batch_size, num_tickers, sequence_length, feature_length))
#     target_tensor = torch.randint(2, (batch_size, num_tickers, 1)).float()
#     output = model(in_tensor)
#     assert output.size() == (batch_size, num_tickers, 1)


# @pytest.mark.parametrize('sequence_length', [3, 5, 7])
# def test_different_sequence_lengths(model_params, sequence_length):
#     model = LSTMModel(**model_params)
#     batch_size = 4
#     num_tickers = 3
#     feature_length = model_params['input_size']

#     in_tensor = torch.rand(
#         (batch_size, num_tickers, sequence_length, feature_length))
#     target_tensor = torch.randint(2, (batch_size, num_tickers, 1)).float()
#     output = model(in_tensor)
#     assert output.size() == (batch_size, num_tickers, 1)


# @pytest.mark.parametrize('feature_length', [5, 10, 15])
# def test_different_feature_lengths(model_params, feature_length):
#     model_params_copy = model_params.copy()
#     model_params_copy['input_size'] = feature_length
#     model = LSTMModel(**model_params_copy)
#     batch_size = 4
#     num_tickers = 3
#     sequence_length = model_params['sequence_length']

#     in_tensor = torch.rand(
#         (batch_size, num_tickers, sequence_length, feature_length))
#     target_tensor = torch.randint(2, (batch_size, num_tickers, 1)).float()
#     output = model(in_tensor)
#     assert output.size() == (batch_size, num_tickers, 1)


# def test_droupout_effect(model_params):
#     model = LSTMModel(**model_params)
#     in_tensor = torch.rand(
#         (4, 3, model_params['sequence_length'], model_params['input_size']))

#     # test in train mode
#     model.train()
#     out1 = model(in_tensor)
#     out2 = model(in_tensor)
#     # output should be different because of dropout
#     assert not torch.allclose(out1, out2)

#     # test in eval model
#     model.eval()
#     out3 = model(in_tensor)
#     out4 = model(in_tensor)
#     # output should be the same because dropout is turned off
#     assert torch.allclose(out3, out4)


# def test_edge_cases_zero_input_size(model_params):
#     model_params_copy = model_params.copy()
#     model_params_copy['input_size'] = 0
#     model = LSTMModel(**model_params_copy)
#     in_tensor = torch.rand((4, 3, model_params_copy['sequence_length'], 0))
#     output = model(in_tensor)
#     assert not torch.isnan(output).any()


# def test_edge_cases_large_input_size(model_params):
#     model_params_copy = model_params.copy()
#     model_params_copy['input_size'] = 1000
#     model = LSTMModel(**model_params_copy)
#     in_tensor = torch.rand(
#         (4, 3, model_params_copy['sequence_length'], 1000))
#     output = model(in_tensor)
#     assert not torch.isnan(output).any()
#     # check if the output is a probability
#     assert torch.all((output >= 0) & (output <= 1))


# def test_learning_rate_override(model_params):
#     model = LSTMModel(**model_params)
#     optimizer = model.configure_optimizers()
#     assert optimizer.param_groups[0]['lr'] == model_params['learning_rate']

#     model_params_copy = model_params.copy()
#     model_params_copy['learning_rate'] = 1e-4
#     model = LSTMModel(**model_params_copy)
#     optimizer = model.configure_optimizers()
#     assert optimizer.param_groups[0]['lr'] == model_params_copy['learning_rate']
