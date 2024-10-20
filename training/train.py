import torch
from models.lstm_model import LSTMModel
from lightning.pytorch import Trainer
from training.train_utils import get_config, get_logger


def main():
    logger = get_logger()
    logger.info(f'Using torch {torch.__version__}')
    config = get_config() # get hyperparameters from config.yaml
    lstm = LSTMModel(config['INPUT_SIZE'], config['HIDDEN_SIZE'])
    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        default_root_dir="checkpoints")
    #train_loader = [] # TODO: add dataloader here
    #trainer.fit(model=lstm, train_dataloaders=train_loader)

if __name__ == '__main__':
    main()