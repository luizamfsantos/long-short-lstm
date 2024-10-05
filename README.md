# long-short-lstm


# Repository structure
|- data
  |-- raw/                         # Store raw, unprocessed data
    |-- market_data.csv
    |-- fundamental_data.csv
  |-- processed/                   # Store preprocessed data (ready for modeling)
    |-- train_data.csv
    |-- test_data.csv
  |-- stock_list.csv               # List of stocks to be used in the model
|- ingestion
  |-- get_data.py                  # Script to retrieve data from API/source
  |-- preprocess.py                # Data preprocessing (cleaning, feature engineering)
  |-- ingestion_utils.py           # Utility functions for data retrieval and handling
|- models
  |-- lstm_model.py                # Define LSTM architecture
  |-- data_preparation.py          # Prepare data for model training
  |-- model_utils.py               # Helper functions for models (loading, saving models, etc.)
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
|- tests                           # Unit tests and validation scripts
  |-- test_ingestion.py            # Test data ingestion functions
  |-- test_model.py                # Test model-related code (training, evaluation, etc.)
  |-- test_trading.py              # Test trading simulation and strategy logic
  |-- test_simulator.py            # Test trading simulator
|- config                          # Configuration files and hyperparameters
  |-- config.yml                   # Store parameters like batch size, learning rate, etc.
  |-- credentials.yml              # Store API keys securely
|- logs                            # Logging for data ingestion, training, and simulation
  |-- ingestion.log
  |-- model_training.log
  |-- trading_simulation.log
|- scripts                         # Additional utility scripts
  |-- run_training.sh              # Shell script to run training
  |-- run_tests.sh                 # Shell script to run tests
  |-- run_simulation.sh            # Shell script to run trading simulation
|- README.md                       # Overview of the project
|- requirements.txt                # Dependencies for the project
|- Dockerfile                      # Dockerfile for containerization
|- docker-compose.yml              # Docker Compose configuration
|- .dockerignore                   # Files to be ignored by Docker
|- .gitignore                      # Files to be ignored by Git
|- template.py                    # Template for new Python scripts