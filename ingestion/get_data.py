import argparse
import json
import logging
import pandas as pd
from typing import Tuple
from pathlib import Path, PosixPath
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc
from ingestion.ingestion_utils import (
    get_config,
    get_fundamentalist_data,
    get_market_data,
    get_stock_list,
)
from ingestion.data_model import (
    FundamentalistApiResponse,
    MarketApiResponse,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def pipeline(
    start_time: str,
    end_time: str,
    stock_list: list[str] | None = None,
    config_path: str | None = None,
    save_raw_data: bool = False,
    path_to_save_raw_data: str | PosixPath = Path(
        __file__).parent.parent / 'data' / 'raw',
    path_to_save_combined_data: str | PosixPath = Path(
        __file__).parent.parent / 'data' / 'raw_combined'
) -> None:
    # Get authentication
    config = get_config(config_path)
    # Get list of stocks being ingested
    stock_list = stock_list or get_stock_list()
    # Get data for each stock
    market_data_list, fundamentalist_data_list = zip(*[
        process_stock(start_time,
                      end_time,
                      ticker,
                      config,
                      save_raw_data,
                      path_to_save_raw_data)
        for ticker in stock_list])
    # Filter out None values
    market_data_list = [data for data in market_data_list if data]
    fundamentalist_data_list = [
        data for data in fundamentalist_data_list if data]
    # Extract data from the API response
    market_data_list = extract_data_from_api_response(market_data_list)
    fundamentalist_data_list = \
        extract_data_from_api_response(fundamentalist_data_list)
    # Combine data by type
    market_data = combine_data(market_data_list)
    fundamentalist_data = combine_data(fundamentalist_data_list)
    # Save combined data
    save_combined_data(
        market_data,
        path_to_save_combined_data,
        'market')
    save_combined_data(
        fundamentalist_data,
        path_to_save_combined_data,
        'fundamentalist')


def process_stock(
    start_time: str,
    end_time: str,
    ticker: str,
    config: dict,
    save_raw_data: bool,
    path_to_save_raw_data: str | PosixPath
) -> Tuple[dict | None, dict | None]:
    logger.info(f'Processing data for {ticker}')
    market_data, fundamentalist_data = process_raw_data(
        start_time,
        end_time,
        ticker,
        config,
        save_raw_data,
        path_to_save_raw_data
    )
    # We need both fundamentalist and market data to proceed
    if market_data and fundamentalist_data:
        market_data = market_data.dict()['tables']  # dict
        market_data['ticker'] = ticker
        fundamentalist_data = fundamentalist_data.dict()['tables']
        fundamentalist_data['ticker'] = ticker
        return market_data, fundamentalist_data
    return None, None


def extract_data_from_api_response(data_list: list[dict]) -> list[dict]:
    """ example: market_data_list = [{'tab0':{'column1':{'2023':'val1','2024':'val2'}},'ticker':'PETR4'},{'tab0':{'column1':{'2023':'val1','2024':'val2'}},'ticker':'VALE3'}] """
    def extract_data(data: dict) -> dict | None:
        ticker = data['ticker']
        stock_data = []
        for k, v in data.items():
            if k != 'ticker':
                stock_data.append(pd.DataFrame(v))
        if not stock_data:
            return None
        stock_df = pd.concat(stock_data)
        if stock_df.empty:
            return None
        stock_df['ticker'] = ticker
        # Clean up column names by removing the stock ticker prefix
        stock_df.columns = stock_df.columns.str.replace(f'{ticker.lower()}_', '')
        return stock_df.to_dict()
    # Extract data from each stock table
    data_list = [extract_data(data) for data in data_list]
    # Filter out None values
    data_list = [data for data in data_list if data]
    return data_list


def combine_data(data_list: list[dict]) -> pd.DataFrame:
    data_list = [pd.DataFrame(data) for data in data_list]
    df = pd.concat([data for data in data_list if not data.empty],
                   join='outer')
    # turn the index into a column
    df = df.reset_index(names='date')
    df['date'] = pd.to_datetime(df['date'])
    return df


def save_combined_data(
    df: pd.DataFrame, 
    path_to_save_combined_data: str | PosixPath, 
    data_type: str
) -> None:
    # transform data into pyarrow table
    # TODO: currently it's transforming everything into strings
    table = pa.Table.from_pandas(df)
    # create year and month columns
    table = table.append_column('year', pc.year(table['date']))
    table = table.append_column('month', pc.month(table['date']))
    # save table to parquet
    now = pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')
    basename = f'{data_type}_data_updated_{now}_' + '{i}.parquet'
    ds.write_dataset(table, 
                    format_path(path_to_save_combined_data),
                     format='parquet', 
                     basename_template=basename,
                     partitioning=['year', 'month'],
                     existing_data_behavior='overwrite_or_ignore')


def process_raw_data(
    start_time: str,
    end_time: str,
    ticker: str,
    config: dict,
    save_raw_data: bool,
    path_to_save_raw_data: str | PosixPath
) -> [MarketApiResponse | None, FundamentalistApiResponse | None]:
    # Get data
    arguments = {'start_time': start_time,
                 'end_time': end_time,
                 'ticker': ticker,
                 'username': config['username'],
                 'password': config['password'],
                 'path_to_save_raw_data': path_to_save_raw_data}
    market_data = get_market_data(**arguments)
    fundamentalist_data = get_fundamentalist_data(**arguments)
    # Ensure data is in the correct format
    try:
        market_data = MarketApiResponse(**market_data)
        fundamentalist_data = FundamentalistApiResponse(**fundamentalist_data)
    except Exception as e:
        logger.error(f'Data for {ticker} is not in the correct format.'
                     f' {arguments=}. Error: {e}')
        return None, None
    # Save raw data
    if save_raw_data:
        arguments.update({'market_data': market_data,
                         'fundamentalist_data': fundamentalist_data})
        save_data_by_stock(**arguments)
    return market_data, fundamentalist_data


def save_data_by_stock(
    ticker: str,
    start_time: str,
    end_time: str,
    path_to_save_raw_data: str | PosixPath,
    market_data: MarketApiResponse,
    fundamentalist_data: FundamentalistApiResponse,
    **kwargs
) -> None:
    def format_time(time: str) -> str:
        return pd.to_datetime(time).strftime('%Y%m%d')

    filename = "{ticker}_{start_time}_{end_time}.json"\
        .format(ticker=ticker,
                start_time=format_time(start_time),
                end_time=format_time(end_time))
    try:
        new_path_to_save_raw_data = format_path(
            path_to_save_raw_data) / filename
        with open(new_path_to_save_raw_data, 'w') as f:
            # doing this way because the serialization of model_dump is not working for timestamp
            json.dump({'market_data': json.loads(
                market_data.model_dump_json()),
                'fundamentalist_data': json.loads(
                fundamentalist_data.model_dump_json())}, f, indent=4)
    except Exception as e:
        logger.error(f'Error saving raw data for {ticker}. {e}')


def format_path(path: str | PosixPath) -> Path:
    if isinstance(path, PosixPath):
        return path
    elif isinstance(path, str):
        return Path(path)
    else:
        raise ValueError('Path not in a valid format')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--start_time',
        type=str,
        required=True,
        help='Start time for the data collection'
    )
    parser.add_argument(
        '-e',
        '--end_time',
        type=str,
        required=True,
        help='End time for the data collection'
    )
    parser.add_argument(
        '-c',
        '--config_path',
        type=str,
        required=False,
        help='Path to the config file'
    )
    parser.add_argument(
        '-sl',
        '--stock_list',
        nargs='+',
        required=False,
        help='List of stock tickers to collect data from'
    )
    parser.add_argument(
        '--save_raw_data',
        action='store_true',
        help='Save raw data'
    )
    parser.add_argument(
        '--path_to_save_raw_data',
        type=str,
        help='Path to save raw data',
        default=Path(__file__).parent.parent / 'data' / 'raw'
    )

    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_arguments()
    pipeline(
        args.start_time,
        args.end_time,
        args.stock_list,
        args.config_path,
        args.save_raw_data,
        args.path_to_save_raw_data
    )


if __name__ == '__main__':
    main()
