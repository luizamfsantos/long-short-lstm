from models.lstm_model import LSTMModel
from lightning.pytorch import Trainer
import torch
import logging

def main():
    logger = logging.getLogger(__name__)
    logger.info(f'Using torch {torch.__version__}')
    lstm = LSTMModel()
    trainer = Trainer(devices="auto", accelerator="auto")
    train_loader = [] # TODO: add dataloader here
    trainer.fit(model=lstm, train_dataloaders=train_loader)

if __name__ == '__main__':
    main()