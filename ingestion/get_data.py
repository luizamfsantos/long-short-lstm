
from ingestion.ingestion_utils import (
    get_config,
    get_fundamentalist_data,
    get_market_data,
    get_stock_list,
)
from ingestion.data_model import (
    MarketApiResponse,
    FundamentalistApiResponse,
)
import pandas as pd
import argparse


def pipeline(
    start_time: str,
    end_time: str,
    stock_list: list[str] | None = None,
    config_path: str | None = None
) -> None:
    config = get_config(config_path)
    stock_list = stock_list or get_stock_list()
    for stock in stock_list:
        market_data = get_market_data(stock, config)
        fundamentalist_data = get_fundamentalist_data(stock, config)
        try:
            market_data = MarketApiResponse(**market_data)
            fundamentalist_data = FundamentalistApiResponse(
                **fundamentalist_data)
        except Exception as e:
            print(f'Error: {e}')
            continue
        # TODO: Store raw data


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--start_time',
        type=str,
        required=True,
        help='Start time for the data collection'
    )
    parser.add_argument(
        '--end_time',
        type=str,
        required=True,
        help='End time for the data collection'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        required=False,
        help='Path to the config file'
    )
    parser.add_argument(
        '--stock_list',
        nargs='+',
        required=False,
        help='List of stock tickers to collect data from'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    pipeline(
        args.start_time,
        args.end_time,
        args.stock_list,
        args.config_path
    )


if __name__ == '__main__':
    main()
