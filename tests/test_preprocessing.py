import pandas as pd
import numpy as np
import pyarrow as pa
import torch
from pathlib import Path
from ingestion.preprocess import (
    read_data,
    DataFramePreprocessor,
    IndexTracker,
    TensorBuilder,
)
import pytest
from pandas.api.types import is_numeric_dtype


@pytest.fixture
def sample_raw_data():
    data_path = 'data/raw_combined/2024/9'
    data_generator = read_data(data_path, batch_size=100)
    return next(data_generator).to_pandas()


@pytest.fixture
def sample_data(sample_raw_data):
    preprocessor = DataFramePreprocessor()
    return preprocessor.process_batch(sample_raw_data)


def test_dataframe_preprocessor(sample_raw_data):
    preprocessor = DataFramePreprocessor()
    processed_df = preprocessor.process_batch(sample_raw_data)
    assert isinstance(processed_df, pd.DataFrame)
    assert not processed_df.empty


def test_index_tracker(sample_data):
    tracker = IndexTracker()
    tracker.update_indices(sample_data)
    assert len(tracker.ticker_idx) > 0
    assert len(tracker.timestamp_idx) > 0
    assert isinstance(tracker.ticker_idx, dict)
    assert isinstance(tracker.timestamp_idx, dict)
    # check if all keys are strings if it's not, print the key
    assert all(isinstance(ticker, str) for ticker in tracker.ticker_idx), [
        ticker for ticker in tracker.ticker_idx if not isinstance(ticker, str)
    ]
    assert all(isinstance(timestamp, pd.Timestamp) for timestamp in tracker.timestamp_idx), [
        timestamp for timestamp in tracker.timestamp_idx if not isinstance(timestamp, pd.Timestamp)
    ]
    assert all(isinstance(idx, int) for idx in tracker.ticker_idx.values()), [
        idx for idx in tracker.ticker_idx.values() if not isinstance(idx, int)
    ]


def test_convert_datatype(sample_data):
    data = sample_data.copy()
    data = DataFramePreprocessor.convert_datatype(data)
    for col in data.columns:
        if col not in ['date', 'ticker']:
            assert is_numeric_dtype(data[col])


def test_calculate_target_variables(sample_data):
    data = sample_data.copy()
    data = DataFramePreprocessor.convert_datatype(data)
    data = DataFramePreprocessor.calculate_target_variable(data)
    assert 'target' in data.columns
    assert is_numeric_dtype(data['target'])
    assert data['target'].nunique() <= 2  # 0 or 1


def test_drop_duplicates(sample_data):
    data = sample_data.copy()
    data = DataFramePreprocessor.convert_datatype(data)
    data = DataFramePreprocessor.calculate_target_variable(data)
    data = DataFramePreprocessor.drop_duplicates(data)
    assert data.duplicated().sum().sum() == 0
