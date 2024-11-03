import os
import torch
import argparse
import shutil
from models.lstm_model import LSTMModel
import torch.nn.functional as F
from lightning.pytorch import Trainer
from training.train_utils import get_config, get_logger
from models.data_preparation import TimeSeriesDataModule
from torch.utils.data import DataLoader
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--clear_checkpoints',
                        action='store_true',
                        help='Clear checkpoint directory before training')
    parser.add_argument('--load_from_checkpoint',
                        type=str,
                        help='Load model from checkpoint used to continue training')
    return parser.parse_args()


def check_checkpoint_valid(checkpoint_file_path: str) -> bool:
    path = Path(checkpoint_file_path)
    return path.is_file() and path.suffix == '.ckpt' and path.stat().st_size > 0

def clear_checkpoints():
    checkpoints_path = Path("checkpoints")
    if checkpoints_path.exists():
        shutil.rmtree(checkpoints_path)
        print("Checkpoints cleared.")

def create_data_loader(config):
    dataset = TimeseriesData(seq_len=config.get('SEQUENCE_LENGTH', 5))
    return DataLoader(dataset, shuffle=False)

def main():
    args = parse_arguments()
    logger = get_logger()
    logger.info(f'Using torch {torch.__version__}')
    config = get_config()  # get hyperparameters from config.yaml
    os.makedirs('checkpoints', exist_ok=True)

    if args.clear_checkpoints:
        clear_checkpoints()

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

    if args.load_from_checkpoint and not check_checkpoint_valid(args.load_from_checkpoint):
        raise ValueError('Invalid checkpoint file path')

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='lstm-{epoch:02d}-{train_loss:.2f}', # TODO: instead of epoch, include date
        monitor='train_loss',
        save_last=True,
        save_top_k=3,
        mode='min'
    )
    trainer = Trainer(
        devices=config.get('DEVICES', 'auto'),
        accelerator=config.get('ACCELERATOR', 'auto'),
        default_root_dir=config.get('ROOT_DIR', 'checkpoints'),
        callbacks=[checkpoint_callback],
        max_epochs=config.get('NUM_EPOCHS', 100),
    )

    data_module = TimeSeriesDataModule(
        data_dir=config.get('DATA_DIR', 'data/processed'),
        batch_size=config.get('BATCH_SIZE', 1),
        seq_len=config.get('SEQUENCE_LENGTH', 5)
    )
    
    if args.load_from_checkpoint:
        trainer.fit(model=lstm, ckpt_path=args.load_from_checkpoint, datamodule=data_module) # TODO: do we need to pass datamodule?
    else:
        trainer.fit(model=lstm, train_dataloaders=train_loader, datamodule=data_module)


if __name__ == '__main__':
    main()
