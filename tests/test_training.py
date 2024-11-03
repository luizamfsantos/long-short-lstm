import pytest
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import shutil
from torch.utils.data import DataLoader
import numpy as np
from tests.test_utils import RandomDataModule
from models.lstm_model import LSTMModel


@pytest.fixture
def model_params():
    return {
        'feature_length': 10,
        'hidden_size': 6,
        'sequence_length': 5,
        'batch_size': 3,
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
        'timestamps': 15,
        'batch_size': 2,
    }


@pytest.fixture
def basic_model(model_params):
    """Create a basic LSTM model instance"""
    return LSTMModel(**model_params)

@pytest.fixture
def basic_data(data_params):
    """Create a basic dataset"""
    return RandomDataModule(**data_params)


def test_model_initialization(basic_model, model_params):
    """ already tested in test_model.py, just to make sure it works
    for the training script as well"""
    assert isinstance(basic_model, LSTMModel)
    assert basic_model.hparams.feature_length == model_params['feature_length']
    assert basic_model.hparams.hidden_size == model_params['hidden_size']
    assert isinstance(basic_model.lstm, torch.nn.LSTM)
    assert isinstance(basic_model.linear, torch.nn.Linear)

def test_checkpoint_saving(basic_model, basic_data, tmp_path):
    ...

def test_checkpoint_loading(basic_model, basic_data, tmp_path):
    # Check if loaded model produces same architecture
#         assert type(loaded_model) == type(basic_model)
#         assert loaded_model.hparams.input_size == basic_model.hparams.input_size
#         assert loaded_model.hparams.hidden_size == basic_model.hparams.hidden_size
    ...

def test_training_manager_initialization():
    ...

def test_training_manager_training():
    ...

def test_training_script():
    ...
