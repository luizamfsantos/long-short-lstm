#!/bin/bash

# Test data ingestion
python -m pytest tests/test_ingestion.py

# Test data preparation
python -m pytest tests/test_data_preparation.py