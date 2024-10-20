#!/bin/python
# this script will be used to preprocess the whole data
# attention that to avoid data leakage, the test data will be normalized 
# using the train data statistics. be careful to not use the test data statistics
# in the training process
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from typing import Iterator
from pathlib import Path
from ingestion.ingestion_utils import get_logger

logger = get_logger('preprocessing')

def calculate_target_variable(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """ The goal of the model is to predict the direction of the stock price.
    Because of this, the target variable will be a binary variable that indicates
    if the stock price will go up or down. This function will calculate the target
    variable based on the stock price."""
    df['target'] = (df[column_name] - df[column_name].shift(1)) > 0

def calculate_returns(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """ Calculate the returns of the stock price. """
    df['returns'] = df[column_name].pct_change()

def read_data(
    folder_path: str | list[str], 
    data_type: str | None = None, 
    batch_size: int | None = 1000
    ) -> Iterator[pa.RecordBatch] | ds.Dataset:
    """ Read the parquet files from the data/raw_combined folder and return a generator 
    to iterate over the data. """
    if isinstance(folder_path, list):
        dataset = ds.dataset(folder_path, format='parquet')
    elif data_type is not None:
        if data_type not in ['market', 'fundamentalist']:
            raise ValueError('data_type should be either market or fundamentalist')
        folder_path = Path(folder_path)
        file_list = [folder_path / file for file in os.listdir(folder_path) if data_type in file]
        if file_list:
            dataset = read_data(file_list, batch_size=None)
        else:
            logging.warning(f'No {data_type} data found in {folder_path}')
            return None
    else:
        market_dataset = read_data(folder_path, data_type='market', batch_size=None)
        fundamentalist_dataset = read_data(folder_path, data_type='fundamentalist', batch_size=None)
        if market_dataset and fundamentalist_dataset:
            dataset = market_dataset.join(fundamentalist_dataset,
                                          keys=['date', 'ticker'],
                                          join_type='full outer')
        else:
            dataset = market_dataset or fundamentalist_dataset
    # sort the data by the timestamp
    dataset = dataset.sort_by('date')
    if batch_size is None:
        return dataset
    return dataset.to_batches(batch_size=batch_size)

# def get_average