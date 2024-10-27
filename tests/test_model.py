import pytest 
import torch 
import torch.nn as nn 
import lightning as L
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, TensorDataset
from models.lstm_model import LSTMModel

@pytest.fixture
def model_params():
    return {
        'input_size': 10,
        'hidden_size': 20,
        'sequence_length': 5,
        'batch_size': 1,
        'num_layers': 2,
        'dropout_rate': 0.2,
        'learning_rate': 1e-3,
        'num_epochs': 100,
    }

@pytest.fixture
def sample_batch():
    batch_size = 4
    num_tickers = 3
    sequence_length = 5
    feature_length = 10

    input_tensor = torch.rand((batch_size, num_tickers, sequence_length, feature_length))
    target_tensor = torch.randint(2, (batch_size, num_tickers, 1)).float() # TODO: do I need the float()?

    return input_tensor, target_tensor

def test_model_init(model_params):
    model = LSTMModel(**model_params)

    # check if the model has the correct attributes
    assert isinstance(model, L.LightningModule)
    assert isinstance(model.lstm, nn.LSTM)
    assert isinstance(model.linear, nn.Linear)

    # check lstm parameters
    assert model.lstm.input_size == model_params['input_size']
    assert model.lstm.hidden_size == model_params['hidden_size']
    assert model.lstm.num_layers == model_params['num_layers']

    # check linear parameters
    assert model.linear.in_features == model_params['hidden_size']
    assert model.linear.out_features == 1

def test_forward_pass_shape(model_params, sample_batch):
    model = LSTMModel(**model_params)
    input_tensor, target_tensor = sample_batch

    output = model(input_tensor)

    batch_size, num_tickers, sequence_length, input_size = input_tensor.size()
    expected_shape = (batch_size, num_tickers, 1)

    assert output.size() == expected_shape
    assert torch.all((output >= 0) & (output <= 1)) # check if the output is a probability

def test_training_step(model_params, sample_batch, tmp_path):
    model = LSTMModel(**model_params)
    test_checkpoint_dir = tmp_path / 'test_checkpoint_dir'
    trainer = Trainer(fast_dev_run=True, default_root_dir=test_checkpoint_dir)
    train_loader = DataLoader(TensorDataset(*sample_batch), batch_size=1, shuffle=False, num_workers=9, persistent_workers=True)
    trainer.fit(model, train_loader)
    # run training step
    loss = model.training_step(sample_batch, batch_idx=0) # TODO: how does batch_idx work?

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert loss.item() >= 0
    assert not torch.isnan(loss) # check if the loss is not NaN
    assert not torch.isinf(loss) # check if the loss is not infinity

def test_test_step(model_params, sample_batch, tmp_path):
    model = LSTMModel(**model_params)
    test_checkpoint_dir = tmp_path / 'test_checkpoint_dir'
    trainer = Trainer(fast_dev_run=True, default_root_dir=test_checkpoint_dir)
    train_loader = DataLoader(TensorDataset(*sample_batch), batch_size=1, shuffle=False, num_workers=9, persistent_workers=True)
    trainer.fit(model, train_loader)
    # run test step
    loss = model.test_step(sample_batch, batch_idx=0) # TODO: how does batch_idx work?

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert loss.item() >= 0
    assert not torch.isnan(loss) # check if the loss is not NaN
    assert not torch.isinf(loss) # check if the loss is not infinity

def test_configure_optimizers(model_params):
    model = LSTMModel(**model_params)
    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Adam)

    for param_group in optimizer.param_groups:
        assert param_group['lr'] == model_params['learning_rate']
        assert param_group['betas'] == (0.9, 0.999)
        assert param_group['eps'] == 1e-8
        assert param_group['weight_decay'] == 0
        assert param_group['amsgrad'] == False
        assert param_group['params'] == list(model.parameters())

@pytest.mark.parametrize('batch_size', [1, 2, 4])
def test_different_batch_sizes(model_params, batch_size):
    model = LSTMModel(**model_params)
    num_tickers = 3
    sequence_length = model_params['sequence_length']
    feature_length = model_params['input_size']

    input_tensor = torch.rand((batch_size, num_tickers, sequence_length, feature_length))
    target_tensor = torch.randint(2, (batch_size, num_tickers, 1)).float()
    output = model(input_tensor)
    assert output.size() == (batch_size, num_tickers, 1)

@pytest.mark.parametrize('num_tickers', [1, 3, 5])
def test_different_ticker_counts(model_params, num_tickers):
    model = LSTMModel(**model_params)
    batch_size = 4
    sequence_length = model_params['sequence_length']
    feature_length = model_params['input_size']

    input_tensor = torch.rand((batch_size, num_tickers, sequence_length, feature_length))
    target_tensor = torch.randint(2, (batch_size, num_tickers, 1)).float()
    output = model(input_tensor)
    assert output.size() == (batch_size, num_tickers, 1)

@pytest.mark.parametrize('sequence_length', [3, 5, 7])
def test_different_sequence_lengths(model_params, sequence_length):
    model = LSTMModel(**model_params)
    batch_size = 4
    num_tickers = 3
    feature_length = model_params['input_size']

    input_tensor = torch.rand((batch_size, num_tickers, sequence_length, feature_length))
    target_tensor = torch.randint(2, (batch_size, num_tickers, 1)).float()
    output = model(input_tensor)
    assert output.size() == (batch_size, num_tickers, 1)

@pytest.mark.parametrize('feature_length', [5, 10, 15])
def test_different_feature_lengths(model_params, feature_length):
    model_params_copy = model_params.copy()
    model_params_copy['input_size'] = feature_length
    model = LSTMModel(**model_params_copy)
    batch_size = 4
    num_tickers = 3
    sequence_length = model_params['sequence_length']

    input_tensor = torch.rand((batch_size, num_tickers, sequence_length, feature_length))
    target_tensor = torch.randint(2, (batch_size, num_tickers, 1)).float()
    output = model(input_tensor)
    assert output.size() == (batch_size, num_tickers, 1)