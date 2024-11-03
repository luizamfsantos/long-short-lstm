#!/bin/bash

# Test data ingestion
python -m pytest tests/test_ingestion.py

# Test data preparation
python -m pytest tests/test_data_preparation.py

# Test model
python -m pytest tests/test_model.py

# Test training
python -m pytest tests/test_training.py