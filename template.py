import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s:%(levelname)s]: %(message)s')

project_name = 'long-short-lstm'

list_of_files = [
    'data/raw/.gitkeep',
    'data/raw_combined/.gitkeep',
    'data/processed/.gitkeep',
    'ingestion/get_data.py',
    'ingestion/preprocess.py',
    'ingestion/ingestion_utils.py',
    'ingestion/data_model.py',
    'models/lstm_model.py',
    'models/data_preparation.py',
    'models/model_utils.py',
    'training/train.py',
    'training/evaluate.py',
    'training/train_utils.py',
    'training/train_config.yaml',
    'trading/long_short_strategy.py',
    'simulator/strategy_interface.py',
    'simulator/strategy_simulator.py',
    'results/.gitkeep',
    'tests/test_ingestion.py',
    'tests/test_model.py',
    'tests/test_trading.py',
    'tests/test_simulator.py',
    'config/config.yaml',
    'config/credentials.yml',
    'logs/.gitkeep',
    'logs/ingestion.log',
    'logs/model_training.log',
    'logs/trading_simulation.log',
    'scripts/run_training.sh',
    'scripts/run_tests.sh',
    'scripts/run_simulation.sh',
    'README.md',
    'requirements.txt',
    'Dockerfile',
    'docker-compose.yml',
    '.dockerignore',
    '.gitignore',
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating directory: {filedir} for file: {filename}')

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            logging.info(f'Creating file: {filepath}')
            f.write('')

    else:
        logging.info(f'File: {filepath} already exists')