# the tensor will have the shape [number_tickers, number_timestamps, feature_vector]
# but dataloader will return the shape [batch_size, number_tickers, sequence_length, feature_vector]
# number_tickers is the number of unique tickers in the dataset
# number_timestamps is the number of unique timestamps in the dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import lightning as L
import os

class LSTMModel(L.LightningModule):

    def __init__(
        self, 
        feature_length: int, 
        hidden_size: int,
        sequence_length: int=5,
        batch_size: int=1,
        num_layers: int=2,
        dropout_rate: float=0.2,
        learning_rate: float=1e-3,
        num_epochs: int=100,
        criterion: nn.Module = F.binary_cross_entropy,
        seed: int = os.environ.get('SEED', 42)):
        """ Initialize LSTM unit 
        
        Args:
        feature_length: number of features in the data
        hidden_size: size of output
        sequence_length: number of time steps in the data
        batch_size: number of samples per batch
        num_layers: number of LSTM layers
        dropout_rate: rate in which to drop connections
        learning_rate: rate at which the model learns
        num_epochs: number of times to iterate over the dataset
        criterion: loss function
        """

        super().__init__()
        self.save_hyperparameters()

        L.seed_everything(seed)
        # TODO: is this stateless or stateful?
        # lstm = long short-term memory unit
        self.lstm = nn.LSTM(input_size=feature_length, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout_rate,
                            bidirectional=False,
                            proj_size = 0,
                            batch_first = True,
                            bias=True)
        # linear = fully connected layer for each ticker's prediction
        self.linear = nn.Linear(hidden_size, 1)
        # activation function: transform the output into a probability
        self.activation = torch.sigmoid

    def forward(self, in_tensor: torch.Tensor):
        """ Forward pass
        in_tensor = tensor with shape (batch_size, num_tickers, sequence_length, feature_length)
        lstm_out = tensor with shape (batch_size, num_tickers, 1) # where 1 is the prediction for each ticker
        """
        batch_size, num_tickers, sequence_length, feature_length = in_tensor.size()

        # Process each ticker separately
        ticker_outputs = []
        for i in range(num_tickers):
            # run the LSTM for each ticker
            lstm_out, _ = self.lstm(in_tensor[:, i, :, :]) # (batch_size, sequence_length, hidden_size)
            # get last output
            last_output = lstm_out[:, -1, :] # (batch_size, hidden_size)
            # get the prediction for the ticker
            linear_out = self.linear(last_output) # (batch_size, 1)
            # transform the output into a probability: remember binary classification!
            prediction = self.activation(linear_out) # (batch_size, 1)
            # add singleton dimension for stacking
            prediction = prediction.unsqueeze(1) # (batch_size, 1, 1)
            ticker_outputs.append(prediction)

        # stack the predictions for all tickers
        output = torch.stack(ticker_outputs, dim=1) # (batch_size, num_tickers, 1)
        return output

    def configure_optimizers(self, learning_rate: float | None = None):
        if not learning_rate:
            learning_rate = self.hparams.learning_rate
        return Adam(self.parameters(), lr=learning_rate)

    def training_step(self, batch: [torch.Tensor, torch.Tensor], batch_idx: int):
        # input_i (batch_size, num_tickers, sequence_length, feature_length)
        # target_i (batch_size, num_tickers, 1)
        input_i, target_i = batch # input_i is the input data, target_i is the target data
        output_i = self.forward(input_i) # (batch_size, num_tickers, 1)
        # calculate the loss across all tickers, compare the output to the last value in the sequence of the target
        loss = self.hparams.criterion(output_i[:, :, -1, :], target_i[:, :, -1, :]) # (batch_size, num_tickers, 1)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: [torch.Tensor, torch.Tensor], batch_idx: int):
        input_i, target_i = batch
        output_i = self.forward(input_i)
        loss = self.hparams.criterion(output_i[:, :, -1, :], target_i[:, :, -1, :])
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
