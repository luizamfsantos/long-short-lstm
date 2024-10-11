import argparse
import json
import sys
import logging
import pandas as pd
from pathlib import Path, PosixPath
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
    path_to_save_raw_data: str | PosixPath = Path(__file__).parent.parent / 'data' / 'raw',
) -> None:
    config = get_config(config_path)
    stock_list = stock_list or get_stock_list()
    for ticker in stock_list:
        # Get data
        arguments = {'start_time': start_time, 'end_time': end_time, 'ticker': ticker, 'username': config['username'], 'password': config['password']}
        market_data = get_market_data(**arguments)
        fundamentalist_data = get_fundamentalist_data(**arguments)
        try:
            # Ensure data is in the correct format
            market_data = MarketApiResponse(**market_data)
            fundamentalist_data = FundamentalistApiResponse(
                **fundamentalist_data)
        except Exception as e:
            logger.error(f'Data for {ticker} is not in the correct format. Error: {e}')
            continue
        # Save raw data
        if save_raw_data:
            filename = "{ticker}_{start_time}_{end_time}.json"\
                .format(ticker=ticker, 
                start_time=pd.to_datetime(start_time).strftime('%Y%m%d'), 
                end_time=pd.to_datetime(end_time).strftime('%Y%m%d'))
            if isinstance(path_to_save_raw_data, PosixPath):
                new_path_to_save_raw_data = path_to_save_raw_data / filename 
            else:
                new_path_to_save_raw_data = Path(path_to_save_raw_data) / filename
            with open(new_path_to_save_raw_data, 'w') as f:
                # doing this way because the serialization of model_dump is not working for timestamp
                json.dump({'market_data': json.loads(market_data.model_dump_json()), 
                'fundamentalist_data': json.loads(fundamentalist_data.model_dump_json())}, f, indent=4)


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
