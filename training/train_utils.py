import yaml
import logging
import os
import shutil
from pathlib import Path
import torch
from models.lstm_model import LSTMModel
from lightning.pytorch import Trainer, loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from models.data_preparation import TimeSeriesDataModule


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)

    def clear_checkpoints(self) -> None:
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @staticmethod
    def check_checkpoint_valid(checkpoint_file_path: str) -> bool:
        path = Path(checkpoint_file_path)
        return path.is_file() and path.suffix == '.ckpt' and path.stat().st_size > 0


class TrainingManager:
    def __init__(self, config: dict):
        self.config = config
        self.checkpoint_manager = None
        self.checkpoint_callback = None
        self.trainer = None
        self._setup_trainer()

    def _setup_trainer(self) -> None:
        checkpoint_dir = self.config.get('CHECKPOINT_DIR', 'checkpoints')
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir
        )
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='lstm-{epoch:02d}-{train_loss:.4f}',
            monitor='train_loss',
            save_last=True,
            mode='min',
            save_top_k=3
        )
        tb_logger = loggers.TensorBoardLogger(save_dir="logs/")
        self.trainer = Trainer(
            max_epochs=self.config.get('NUM_EPOCHS', 100),
            devices=self.config.get('DEVICES', 'auto'),
            accelerator=self.config.get('ACCELERATOR', 'auto'),
            callbacks=[self.checkpoint_callback],
            default_root_dir=checkpoint_dir,
            enable_progress_bar=True,
            logger=tb_logger
        )

    def train(
        self,
        model: LSTMModel,
        data_module: TimeSeriesDataModule,
        checkpoint_path: str = None,
    ) -> None:
        if checkpoint_path:
            if not self.checkpoint_manager.check_checkpoint_valid(checkpoint_path):
                raise ValueError('Invalid checkpoint file path')
            self.trainer.fit(
                model=model, ckpt_path=checkpoint_path, datamodule=data_module)
        else:
            self.trainer.fit(model=model, datamodule=data_module)


def get_logger(logger_name='model_training'):
    # Ensure the logs directory exists
    os.makedirs('logs', exist_ok=True)

    # Create a custom logger
    logger = logging.getLogger(logger_name)

    # Set the default logging level
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('logs/model_training.log')

    # Create formatters and add them to handlers
    c_format = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s [[%(filename)s:%(lineno)d]]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    f_format = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s [[%(filename)s:%(lineno)d]]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def get_config():
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    return config['LSTM']
