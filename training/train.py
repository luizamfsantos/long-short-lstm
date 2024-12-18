import os
import torch
import argparse
from typing import Tuple
from models.lstm_model import LSTMModel
import torch.nn.functional as F
from lightning.pytorch import Trainer
from training.train_utils import (
    get_config,
    get_logger,
    TrainingManager,
)
from models.data_preparation import TimeSeriesDataModule


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--clear_checkpoints',
                        action='store_true',
                        help='Clear checkpoint directory before training')
    parser.add_argument('--load_from_checkpoint',
                        type=str,
                        help='Load model from checkpoint used to continue training')
    return parser.parse_args()


def setup_training(config: dict) -> Tuple[LSTMModel, TimeSeriesDataModule, Trainer]:
    model = LSTMModel(
        feature_length=config['FEATURE_LENGTH'],
        hidden_size=config['HIDDEN_SIZE'],
        sequence_length=config.get('SEQUENCE_LENGTH', 5),
        batch_size=config.get('BATCH_SIZE', 1),
        num_layers=config.get('NUM_LAYERS', 2),
        dropout_rate=config.get('DROPOUT_RATE', 0.2),
        learning_rate=config.get('LEARNING_RATE', 1e-3),
        num_epochs=config.get('NUM_EPOCHS', 100),
        criterion=config.get('CRITERION', F.binary_cross_entropy)
    )
    data_module = TimeSeriesDataModule(
        batch_size=config.get('BATCH_SIZE', 1),
        sequence_length=config.get('SEQUENCE_LENGTH', 5),
    )
    return model, data_module


def main():
    # parse arguments
    args = parse_arguments()
    # get logger
    logger = get_logger()
    logger.info(f'Using torch {torch.__version__}')
    # get hyperparameters from config.yaml
    config = get_config()
    # set seed
    if not os.environ.get('SEED'):
        os.environ['SEED'] = str(config.get('SEED', 42))
    # setup training
    lstm, data_module = setup_training(config)
    training_manager = TrainingManager(config)
    if args.clear_checkpoints:
        training_manager.checkpoint_manager.clear_checkpoints()
    logger.info(f'Model initialized with hyperparameters: {lstm.hparams}')
    # start training
    training_manager.train(
        model=lstm,
        data_module=data_module,
        checkpoint_path=args.load_from_checkpoint
    )


if __name__ == '__main__':
    main()
