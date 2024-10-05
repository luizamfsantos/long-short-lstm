import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.optim import Adam
import lightning as L
from torch.utils.data import (
    TensorDataset,
    DataLoader)

SEED: int = 42

class LSTMModel(L.LightningModule):

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int,
        sequence_length: int=5,
        batch_size: int=1,
        num_layers: int=2,
        dropout_rate: float=0.2,
        learning_rate: float=1e-3,
        num_epochs: int=100,
        criterion: nn.Module = f.binary_cross_entropy):
        """ Initialize LSTM unit """

        super(LSTMModel, self).__init__()

        L.seed_everything(seed=SEED)

        # input_size = number of features in the data
        self.input_size=input_size # number of features in the data
        # hidden_size = size of output
        self.hidden_size=hidden_size
        # sequence_length = number of time steps in the data
        self.sequence_length=sequence_length
        # batch_size = number of samples per batch
        self.batch_size=batch_size
        # num_layers = number of LSTM layers
        self.num_layers=num_layers
        # dropout_rate = rate in which to drop connections
        self.dropout_rate=dropout_rate
        # learning_rate = rate at which the model learns
        self.learning_rate=learning_rate
        # num_epochs = number of times to iterate over the dataset
        self.num_epochs=num_epochs
        # criterion = loss function
        self.criterion=criterion
        # lstm = long short-term memory unit
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout_rate,
                            batch_first=True)
        # linear = fully connected layer
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input: torch.Tensor):
        """ Forward pass
        lstm_out = (batch_size, sequence_length, hidden_size)
        """
        lstm_out, _ = self.lstm(input)
        # lstm_out has the short-term memories for all inputs. We make a prediction based on the last one.
        prediction = self.linear(lstm_out[:, -1, :])
        return prediction

    def configure_optimizers(self, learning_rate: float | None = None):
        if not learning_rate:
            learning_rate = self.learning_rate
        return Adam(self.parameters(), lr=learning_rate)

    def training_step(self, batch: [torch.Tensor, torch.Tensor], batch_idx: int):
        input_i, label_i = batch # input_i is the input data, label_i is the target data
        output_i = self.forward(input_i)
        loss = self.criterion(torch.sigmoid(output_i), label_i)
        result = L.TrainResult(loss)
        result.log('train_loss', loss)
        return result

    def validation_step(self, batch: [torch.Tensor, torch.Tensor], batch_idx: int):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self.criterion(torch.sigmoid(output_i), label_i)
        result = L.EvalResult(loss)
        result.log('val_loss', loss)
        return result

    def test_step(self, batch: [torch.Tensor, torch.Tensor], batch_idx: int):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self.criterion(torch.sigmoid(output_i), label_i)
        result = L.EvalResult(loss)
        result.log('test_loss', loss)
        return result
