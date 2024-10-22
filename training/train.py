import torch
from models.lstm_model import LSTMModel
import torch.nn.functional as F
from lightning.pytorch import Trainer
from training.train_utils import get_config, get_logger
from models.data_preparation import TimeSeriesData


def main():
    logger = get_logger()
    logger.info(f'Using torch {torch.__version__}')
    config = get_config() # get hyperparameters from config.yaml
    lstm = LSTMModel(
        input_size=config['INPUT_SIZE'], 
        hidden_size=config['HIDDEN_SIZE'],
        sequence_length=config.get('SEQUENCE_LENGTH', 5),
        batch_size=config.get('BATCH_SIZE', 1),
        num_layers=config.get('NUM_LAYERS', 2),
        dropout_rate=config.get('DROPOUT_RATE', 0.2),
        learning_rate=config.get('LEARNING_RATE', 1e-3),
        num_epochs=config.get('NUM_EPOCHS', 100),
        criterion=config.get('CRITERION', F.binary_cross_entropy))
    logger.info(f'Model initialized with hyperparameters: {lstm.hparams}')
    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        default_root_dir="checkpoints")
    train_dataset = TimeseriesDataset(seq_len = config.get('SEQUENCE_LENGTH', 5))
    train_loader = DataLoader(train_dataset, shuffle = False)
    trainer.fit(model=lstm, train_dataloaders=train_loader)

if __name__ == '__main__':
    main()