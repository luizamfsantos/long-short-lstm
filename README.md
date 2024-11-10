# Long Short Trader

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Repository structure](#repository-structure)

## Introduction
Long Short Trader is a project that aims to build a trading strategy using deep learning models. The project uses a Long Short Term Memory (LSTM) neural network to predict returns direction and generate trading signals. The trading strategy is implemented using a long-short approach, where the model predicts the direction of the stock price movement and takes a long position if the price is expected to increase and a short position if the price is expected to decrease.

## Features
- Data ingestion: Retrieve historical stock data from an API and preprocess it for modeling
- LSTM model: Implement a deep learning model using LSTM architecture to predict stock prices
- Trading strategy: Implement a long-short trading strategy based on the model predictions
- Simulation: Simulate the trading strategy on historical data to assess its effectiveness

## Installation
1. Clone the repository:
```bash
git clone https://github.com/luizamfsantos/long-short-lstm.git
```

2. Add your credentials to the `config/credentials.yml` file:
```yaml
username: your_username
password: your_password
```

3. *[Optional]* Modify the `config/config.yml` file to adjust the hyperparameters for training the model.


4. Build the Docker container:
```bash
docker build -t long-short-trader .
```


## Usage 
1. Run the Docker container:
```bash
docker run -it long-short-trader
```
2. To run the ingestion script:
```bash
./scripts/run_ingestion.sh
```

3. To train the model:
```bash
./scripts/run_training.sh
```

4. To run the trading simulation:
```bash
./scripts/run_simulation.sh
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)



## Repository structure
```{lua}
|- checkpoints                     # Store model checkpoints
  |-- model_20241001.ckpt
  |-- model_20241014.ckpt
|- data
  |-- raw/                         # Store raw, unprocessed data by stock
    |-- PETR4.json
    |-- VALE3.json
  |-- raw_combined/                # Store raw, unprocessed data by date
    |-- market_data_2023.parquet
    |-- market_data_2024.parquet
    |-- fundamental_data_2023.parquet
    |-- fundamental_data_2024.parquet
  |-- processed/                   # Store preprocessed data (ready for modeling)
    |-- train_data.parquet
    |-- test_data.parquet
  |-- stock_list.csv               # List of stocks to be used in the model
|- ingestion
  |-- get_data.py                  # Script to retrieve data from API/source
  |-- preprocess.py                # Data preprocessing (cleaning, feature engineering)
  |-- ingestion_utils.py           # Utility functions for data retrieval and handling
|- models
  |-- lstm_model.py                # Define LSTM architecture
  |-- data_preparation.py          # Prepare data for model training
  |-- model_utils.py               # Helper functions for models (loading, saving models, etc.)
|- libs
  |-- quantstats-0.0.63            # Modified version of QuantStats library
|- training
  |-- train.py                     # Train the model
  |-- evaluate.py                  # Evaluate model performance (backtesting, validation)
  |-- train_utils.py               # Utility functions for training and evaluation
  |-- train_config.yml             # Configuration file for training hyperparameters
|- trading
  |-- long_short_strategy.py       # Long-short trading strategy logic
|- simulator
    |-- strategy_interface.py      # Interface for trading strategies
    |-- strategy_simulator.py      # Trading simulator
|- results
  |-- ret_port.parquet             # Store returns of the portfolio
  |-- weights_db.parquet           # Store weights of the portfolio
|- tests                           # Unit tests and validation scripts
  |-- test_ingestion.py            # Test data ingestion functions
  |-- test_model.py                # Test model-related code (training, evaluation, etc.)
  |-- test_trading.py              # Test trading simulation and strategy logic
  |-- test_simulator.py            # Test trading simulator
|- config                          # Configuration files and hyperparameters
  |-- config.yml                   # Store parameters like batch size, learning rate, etc.
  |-- credentials.yml              # Store authentication credentials
|- logs                            # Logging for data ingestion, training, and simulation
  |-- ingestion.log
  |-- model_training.log
  |-- trading_simulation.log
|- scripts                         # Additional utility scripts  
  |-- run_ingestion.sh             # Shell script to run data ingestion
  |-- run_training.sh              # Shell script to run training
  |-- run_tests.sh                 # Shell script to run tests (for CI/CD)
  |-- run_simulation.sh            # Shell script to run trading simulation
|- README.md                       # Overview of the project
|- requirements.txt                # Dependencies for the project
|- Dockerfile                      # Dockerfile for containerization
|- docker-compose.yml              # Docker Compose configuration
|- .dockerignore                   # Files to be ignored by Docker
|- .gitignore                      # Files to be ignored by Git
|- template.py                     # Template for new Python scripts
```
