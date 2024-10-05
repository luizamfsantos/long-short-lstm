import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import lightning as L
import os

SEED: int = int(os.getenv('GLOBAL_SEED', L.seed_everything()))

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
        criterion: nn.Module = F.binary_cross_entropy):
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
                            bidirectional=False,
                            proj_size = 0,
                            batch_first = True,
                            bias=True)
        # linear = fully connected layer
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input: torch.Tensor):
        """ Forward pass
        input = tensor with shape (batch_size, sequence_length, input_size)
        lstm_out = tensor with shape (batch_size, sequence_length, hidden_size)
        """
        lstm_out, _ = self.lstm(input)
        # lstm_out has the short-term memories for all inputs. We make a prediction based on the last one.
        linear_out = self.linear(lstm_out[:, -1, :])
        # transform solution in probability
        prediction = torch.sigmoid(linear_out)
        return prediction

    def configure_optimizers(self, learning_rate: float | None = None):
        if not learning_rate:
            learning_rate = self.learning_rate
        return Adam(self.parameters(), lr=learning_rate)

    def training_step(self, batch: [torch.Tensor, torch.Tensor], batch_idx: int):
        input_i, target_i = batch # input_i is the input data, target_i is the target data
        output_i = self.forward(input_i)
        loss = self.criterion(output_i, target_i)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: [torch.Tensor, torch.Tensor], batch_idx: int):
        input_i, target_i = batch
        output_i = self.forward(input_i)
        loss = self.criterion(output_i, target_i)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: [torch.Tensor, torch.Tensor], batch_idx: int):
        input_i, target_i = batch
        output_i = self.forward(input_i)
        loss = self.criterion(output_i, target_i)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
