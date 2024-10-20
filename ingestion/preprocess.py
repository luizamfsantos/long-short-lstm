#!/bin/python
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

def read_data(
    folder_path: str | list[str], 
    data_type: str | None = None, 
    batch_size: int | None = 1000
    ) -> Iterator[pa.RecordBatch] | ds.Dataset | None:
    """ Read the parquet files from the data/raw_combined folder and return a generator 
    to iterate over the data. """
    if isinstance(folder_path, list):
        dataset = ds.dataset(folder_path, format='parquet')
    elif data_type is not None:
        if data_type not in ['market', 'fundamentalist']:
            raise ValueError('data_type should be either market or fundamentalist')
        folder_path = Path(folder_path)
        file_list = list(folder_path.glob(f'**/*{data_type}*.parquet'))
        if file_list:
            dataset = read_data(file_list, batch_size=None)
        else:
            logger.warning(f'No {data_type} data found in {folder_path}')
            return None
    else:
        market_dataset = read_data(folder_path, data_type='market', batch_size=None)
        fundamentalist_dataset = read_data(folder_path, data_type='fundamentalist', batch_size=None)
        # check that the datasets have the data and ticker columns
        if 'date' not in market_dataset.schema.names or 'ticker' not in market_dataset.schema.names:
            market_dataset = None
        if 'date' not in fundamentalist_dataset.schema.names or 'ticker' not in fundamentalist_dataset.schema.names:
            fundamentalist_dataset = None  
        if market_dataset and fundamentalist_dataset:
            dataset = market_dataset.join(fundamentalist_dataset,
                                          keys=['date', 'ticker'],
                                          join_type='full outer')
        else:
            dataset = market_dataset or fundamentalist_dataset
            if not dataset:
                logger.warning(f'No data found in {folder_path}')
                return None
    # sort the data by the timestamp
    dataset = dataset.sort_by('date')
    if batch_size is None:
        return dataset
    return dataset.to_batches(batch_size=batch_size)

def convert_datatype(df: pd.DataFrame) -> pd.DataFrame:
    """ Convert the data types of the columns.
    Ticker should be object,
    Date should be datetime,
    Everything else should be float64 """
    df['ticker'] = df['ticker'].astype('object')
    df['date'] = pd.to_datetime(df['date'])
    for column in df.columns:
        if column not in ['ticker', 'date']:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(how='all', axis=1)
    return df

def preprocess(
    input_path: str | list[str],
    output_path: str,
    test: bool = False):
    # Get data
    data = read_data(input_path) # TODO: modify to read in batches
    # Convert prices to returns
    calculate_returns(data, 'price') # TODO: change the column name
    # Calculate the mean and standard deviation of the train data
    if not test:
        train_stats = get_train_stats(data)
        # TODO: Save the train data statistic
    else:
        train_stats = ... # TODO: Read the train data statistics
    # Normalize the data
    data_normalized = normalize_data(data, train_stats)
    # TODO: Save the normalized data



