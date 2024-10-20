#!/bin/python
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from typing import Iterator
from pathlib import Path
import torch
from ingestion.ingestion_utils import get_logger

# Step 1: Sort the dataframe by both date and ticker so that we can easily map values to the tensor.
# Step 2: Create a dictionary of tickers and their corresponding indices.
# Step 3: Iterate through the dataframe and create a tensor where each row corresponds to a ticker and each column represents the features for a specific timestamp.
# Step 4: Create a tensor for the target variable.
# the tensor X will have the shape [number_tickers, number_timestamps, feature_vector] 
# the tensor y will have the shape [number_tickers, number_timestamps, 1]

logger = get_logger('preprocessing')

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

def calculate_target_variable(df: pd.DataFrame, return_column: str = 'variacaopercent') -> None:
    """ The goal of the model is to predict the direction of the stock price.
    Because of this, the target variable will be a binary variable that indicates
    if the stock price will go up. This function will calculate the target
    variable based on returns."""
    df['target'] = df[return_column] > 0
    df['target'] = df['target'].astype('int8')


def convert_to_tensor(df: pd.DataFrame) -> tuple:
    """ Convert the dataframe to a tensor.
    The tensor will have the shape [number_tickers, number_timestamps, feature_vector] """
    # Step 1: Sort the dataframe by both date and ticker so that we can easily map values to the tensor.
    df = df.sort_values(by=['date', 'ticker'])
    # Step 2: Create a dictionary of tickers and their corresponding indices.
    tickers = df['ticker'].unique()
    ticker_indices = {ticker: i for i, ticker in enumerate(tickers)}
    # Step 3: Iterate through the dataframe 
    # and create a tensor where each row corresponds to a ticker 
    # and each column represents the features for a specific timestamp.
    tensor = []
    for date, group in df.groupby('date'):
        tensor_row = []
        for _, row in group.iterrows():
            ticker_index = ticker_indices[row['ticker']]
            tensor_row.append(row.drop(['date', 'ticker']).values)
        tensor.append(tensor_row)
    # Step 4: Create a tensor for the target variable.
    target = df['target'].values
    return tensor, target


def convert_to_tensor_v2(data_generator: Iterator[pa.RecordBatch]) -> tuple:
    """ Convert the data generator to a tensor.
    The tensor will have the shape [number_tickers, number_timestamps, feature_vector] """
    # Step 1: Sort the dataframe by both date and ticker so that we can easily map values to the tensor.
    # Step 2: Create a dictionary of tickers and their corresponding indices.
    # Step 3: Iterate through the dataframe and create a tensor where each row corresponds to a ticker and each column represents the features for a specific timestamp.
    # Step 4: Create a tensor for the target variable.
    # Initialize empty dictionaries to track unique tickers and timestamps
    ticker_idx = {}
    timestamp_idx = {}
    # Set an initial index counter for both tickers and timestamps
    ticker_counter = 0
    timestamp_counter = 0
    for batch in data_generator:
        batch_df = batch.to_pandas()
        # Iterate through the rows in the batch
        for _, row in batch_df.iterrows():
            # Check if the ticker is already in the dictionary
            if row['ticker'] not in ticker_idx:
                # If not, add the ticker to the dictionary and increment the counter
                ticker_idx[row['ticker']] = ticker_counter
                ticker_counter += 1
            # Check if the timestamp is already in the dictionary
            if row['date'] not in timestamp_idx:
                # If not, add the timestamp to the dictionary and increment the counter
                timestamp_idx[row['date']] = timestamp_counter
                timestamp_counter += 1
        # Initialize an empty tensor to store the data
        tensor = torch.zeros(ticker_counter, timestamp_counter, len(batch_df.columns) - 2)
        # Initialize an empty tensor to store the target variable
        target = torch.zeros(ticker_counter, timestamp_counter, 1)
        # Iterate through the rows in the batch
        for _, row in batch_df.iterrows():
            # Get the index of the ticker and timestamp
            ticker_index = ticker_idx[row['ticker']]
            timestamp_index = timestamp_idx[row['date']]
            # Add the data to the tensor
            tensor[ticker_index, timestamp_index, :] = torch.tensor(row.drop(['date', 'ticker', 'target']).values)
            # Add the target variable to the target tensor
            target[ticker_index, timestamp_index, 0] = row['target']
        # Save the tensors to disk
        torch.save(tensor, 'data/tensor.pt')
        torch.save(target, 'data/target.pt')
    return tensor, target


def convert_to_tensor_v4(data_generator: Iterator[pa.RecordBatch], batch_size: int = 1000) -> tuple:
    """ Convert the data generator to tensors in batches, 
    saving intermediate results to disk.
    Returns the paths to the saved tensor files. """
    ticker_idx = {}
    timestamp_idx = {}
    feature_names = None
    batch_counter = 0
    tensor_path = 'data/tensor_batches'
    target_path = 'data/target_batches'
    os.makedirs(tensor_path, exist_ok=True)
    os.makedirs(target_path, exist_ok=True)
    for batch in data_generator:
        batch_df = batch.to_pandas()
        if feature_names is None:
            feature_names = [col for col in batch_df.columns if col not in ['date', 'ticker', 'target']]
        for _, row in batch_df.iterrows():
            ticker = row['ticker']
            timestamp = row['date']
            if ticker not in ticker_idx:
                ticker_idx[ticker] = len(ticker_idx)
            if timestamp not in timestamp_idx:
                timestamp_idx[timestamp] = len(timestamp_idx)
        
        # Create batch tensors
        batch_tensor = torch.zeros(len(ticker_idx), len(timestamp_idx), len(feature_names))
        batch_target = torch.zeros(len(ticker_idx), len(timestamp_idx), 1)
        for _, row in batch_df.iterrows():
            ticker_index = ticker_idx[row['ticker']]
            timestamp_index = timestamp_idx[row['date']]
            batch_tensor[ticker_index, timestamp_index, :] = torch.tensor(row[feature_names].values)
            batch_target[ticker_index, timestamp_index, 0] = row['target']
        # Save tensors to disk
        torch.save(batch_tensor, f'{tensor_path}/tensor_{batch_counter}.pt')
        torch.save(batch_target, f'{target_path}/target_{batch_counter}.pt')
        batch_counter += 1

        # Save metadata to disk
        metadata = {
            'ticker_idx': ticker_idx,
            'timestamp_idx': timestamp_idx,
            'feature_names': feature_names,
            'num_batches': batch_counter
        }
        torch.save(metadata, 'data/metadata.pt')

        # Clear memory
        del batch_tensor
        del batch_target
        del batch_df
    return tensor_path, target_path