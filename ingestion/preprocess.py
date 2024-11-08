#!/bin/python
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from typing import Iterator, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
from ingestion.ingestion_utils import get_logger

# Step 1: Sort the dataframe by both date and ticker so that we can easily map
#            values to the tensor.
# Step 2: Create a dictionary of tickers and their corresponding indices.
# Step 3: Iterate through the dataframe and create a tensor where each row corresponds 
#           to a ticker and each column represents the features for a specific timestamp.
# Step 4: Create a tensor for the target variable.
# the tensor X will have the shape [number_tickers, number_timestamps, feature_vector]
# the tensor y will have the shape [number_tickers, number_timestamps, 1]

logger = get_logger('preprocessing')

@dataclass
class TensorMetadata:
    ticker_idx: dict[str, int]
    timestamp_idx: dict[str, int]
    feature_names: list[str]
    num_batches: int

    def save(self, path: str) -> None:
        torch.save(self, path)

    @classmethod
    def load(cls, path: str) -> 'TensorMetadata':
        data = torch.load(path, weights_only=False)
        return cls(**data)


@dataclass
class BatchData:
    tensor: torch.Tensor
    target: torch.Tensor
    metadata: TensorMetadata

class DataFramePreprocessor:
    @staticmethod
    def convert_datatype(df: pd.DataFrame) -> pd.DataFrame:
        """ Convert the data types of the columns.
        Ticker should be string,
        Date should be datetime,
        Everything else should be float64 """
        df['ticker'] = df['ticker'].astype('str')
        df['date'] = pd.to_datetime(df['date'])
        for column in df.columns:
            if column not in ['ticker', 'date']:
                df[column] = pd.to_numeric(df[column], errors='coerce')
        df = df.dropna(how='all', axis=1)
        return df

    # TODO: fix this function
    # @staticmethod
    # def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    #     """ Deal with missing values in the dataframe.
    #     For now, we will forward fill the missing values. """
    #     return df.groupby('ticker').apply(lambda group: group.ffill()).reset_index(drop=False)

    @staticmethod
    def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """ Drop duplicates from the dataframe. """
        return df.drop_duplicates()

    @staticmethod
    def calculate_target_variable(df: pd.DataFrame, return_column: str = 'variacaopercent') -> pd.DataFrame:
        """ The goal of the model is to predict the direction of the stock price.
        Because of this, the target variable will be a binary variable that indicates
        if the stock price will go up. This function will calculate the target
        variable based on returns."""
        df['target'] = df[return_column].gt(0).astype(float)
        return df

    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Process a batch of data. """
        df = self.convert_datatype(df)
        df = self.drop_duplicates(df)
        df = self.calculate_target_variable(df)
        # df = self.handle_missing_values(df)
        return df

class IndexTracker:
    def __init__(self, ticker_idx: dict[str, int] = None, timestamp_idx: dict[pd.Timestamp, int] = None):
        if ticker_idx is None:
            self.ticker_idx: dict[str, int] = {}
        else:
            self.ticker_idx: dict[str, int] = ticker_idx
        if timestamp_idx is None:
            self.timestamp_idx: dict[pd.Timestamp, int] = {}
        else:
            self.timestamp_idx: dict[pd.Timestamp, int] = timestamp_idx

    def update_indices(self, df: pd.DataFrame) -> None:
        for ticker in df['ticker'].unique().tolist():
            if ticker not in self.ticker_idx:
                self.ticker_idx[ticker] = len(self.ticker_idx)

        for timestamp in df['date'].unique():
            if timestamp not in self.timestamp_idx:
                self.timestamp_idx[timestamp] = len(self.timestamp_idx)

class TensorBuilder:
    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names

    def build_tensors(
        self,
        df: pd.DataFrame,
        ticker_idx: dict[str, int],
        timestamp_idx: dict[pd.Timestamp, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_tensor = torch.zeros(len(ticker_idx),
                                   len(timestamp_idx),
                                   len(self.feature_names))
        batch_target = torch.zeros(len(ticker_idx),
                                   len(timestamp_idx),
                                   1)
        for row in df.itertuples(index=False):
            ticker_index = ticker_idx[row.ticker]
            timestamp_index = timestamp_idx[row.date]
            batch_tensor[ticker_index, timestamp_index, :] = torch.tensor(
                [getattr(row, feature) for feature in self.feature_names])
            batch_target[ticker_index, timestamp_index, 0] = row.target
        return batch_tensor, batch_target

class TensorConverter:
    EXCLUDED_COLUMNS = ['date', 'ticker', 'target', 'p/l', 
    'despesa_de_depreciacao,_amortizacao_e_exaustao_3_meses_consolidado__milhoes']

    def __init__(
        self,
        tensor_path: str = 'data/processed/tensor_batches',
        target_path: str = 'data/processed/target_batches',
        metadata_path: str = 'data/processed/metadata.pt',
    ):
        self.tensor_path = tensor_path
        self.target_path = target_path
        self.metadata_path = metadata_path
        self.preprocessor = DataFramePreprocessor()
        self.index_tracker = IndexTracker()
        self.tensor_builder: TensorBuilder = None
        self.batch_counter = 0

        os.makedirs(tensor_path, exist_ok=True)
        os.makedirs(target_path, exist_ok=True)

    def _get_feature_names(self, df: pd.DataFrame) -> list[str]:
        return [col for col in df.columns if col not in self.EXCLUDED_COLUMNS]

    def _save_batch_tensors(
        self, 
        batch_tensor: torch.Tensor, 
        batch_target: torch.Tensor
    ) -> None:
        torch.save(batch_tensor, f'{self.tensor_path}/tensor_{self.batch_counter}.pt')
        torch.save(batch_target, f'{self.target_path}/target_{self.batch_counter}.pt')
        self.batch_counter += 1

    def _save_metadata(self) -> None:
        metadata = TensorMetadata(
            ticker_idx=self.index_tracker.ticker_idx,
            timestamp_idx=self.index_tracker.timestamp_idx,
            feature_names=self.tensor_builder.feature_names,
            num_batches=self.batch_counter
        )
        metadata.save(self.metadata_path)

    def _log_summary(self) -> None:
        logger.info(
            f'Number of batches: {self.batch_counter}\n'
            f'{len(self.index_tracker.ticker_idx)} tickers found: '
            f'{list(self.index_tracker.ticker_idx.keys())}\n'
            f'{len(self.index_tracker.timestamp_idx)} timestamps found.\n'
            f'Min and max timestamps: '
            f'{min(self.index_tracker.timestamp_idx.keys())} '
            f'and {max(self.index_tracker.timestamp_idx.keys())}'
        )

    def process_batch(self, batch: pa.RecordBatch) -> None:
        df = batch.to_pandas()
        df = self.preprocessor.process_batch(df)

        if self.tensor_builder is None:
            feature_names = self._get_feature_names(df)
            self.tensor_builder = TensorBuilder(feature_names)

        self.index_tracker.update_indices(df)
        batch_tensor, batch_target = self.tensor_builder.build_tensors(
            df,
            self.index_tracker.ticker_idx,
            self.index_tracker.timestamp_idx
        )
        self._save_batch_tensors(batch_tensor, batch_target)

    def convert_to_tensor(
        self,
        data_generator: Iterator[pa.RecordBatch]
    ) -> None:
        for batch in data_generator:
            self.process_batch(batch)
        self._save_metadata()
        self._log_summary()


def read_data(
    folder_path: str | list[str],
    data_type: str | None = None,
    batch_size: int | None = 1000
) -> Iterator[pa.RecordBatch] | ds.Dataset | None:
    """ Read the parquet files from the data/raw_combined 
    folder and return a generator to iterate over the data. """
    if isinstance(folder_path, list):
        dataset = ds.dataset(folder_path, format='parquet')
    elif data_type is not None:
        if data_type not in ['market', 'fundamentalist']:
            raise ValueError('data_type should be either '
                             'market or fundamentalist')
        folder_path = Path(folder_path)
        file_list = list(folder_path.glob(f'**/*{data_type}*.parquet'))
        if file_list:
            dataset = read_data(file_list, batch_size=None) # recursive call
        else:
            logger.warning(f'No {data_type} data found in {folder_path}')
            return None
    else:
        market_dataset = read_data(folder_path,
                                   data_type='market',
                                   batch_size=None)
        fundamentalist_dataset = read_data(folder_path,
                                           data_type='fundamentalist',
                                           batch_size=None)
        # check that the datasets have the data and ticker columns
        if 'date' not in market_dataset.schema.names \
                or 'ticker' not in market_dataset.schema.names:
            market_dataset = None
        if 'date' not in fundamentalist_dataset.schema.names \
                or 'ticker' not in fundamentalist_dataset.schema.names:
            fundamentalist_dataset = None
        # combine schemas
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


def load_tensors(
    tensor_path: str = 'data/processed/tensor_batches',
    target_path: str = 'data/processed/target_batches',
    metadata_path: str = 'data/processed/metadata.pt',
    batches: bool = True,
) -> Iterator[BatchData] | BatchData:
    """ Load the tensors from disk and return a generator to iterate over them. """
    metadata = torch.load('data/processed/metadata.pt', weights_only=False)
    if batches:
        for i in range(metadata.num_batches):
            tensor = torch.load(f'{tensor_path}/tensor_{i}.pt')
            target = torch.load(f'{target_path}/target_{i}.pt')
            yield BatchData(tensor=tensor, target=target, metadata=metadata)
    else:
        tensor = torch.load(f'{tensor_path}')
        target = torch.load(f'{target_path}')
        return BatchData(tensor=tensor, target=target, metadata=metadata)


def preprocess_data(
    raw_data_path: str = 'data/raw_combined',
    preprocessed_tensor_path: str = 'data/processed/tensor_batches',
    preprocessed_target_path: str = 'data/processed/target_batches',
    batch_size: int = 1000
) -> None:
    """ Preprocess the data and save the tensors to disk. """
    data_generator = read_data(raw_data_path, batch_size=batch_size)
    logger.info('Data read successfully')
    convert_to_tensor(data_generator,
                      tensor_path=preprocessed_tensor_path,
                      target_path=preprocessed_target_path)
    logger.info('Tensors saved successfully. They can be found in %s and %s',
                preprocessed_tensor_path, preprocessed_target_path)


if __name__ == '__main__':
    preprocess_data()
