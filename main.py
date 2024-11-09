import argparse
import pandas as pd
import quantstats as qs
from trading.long_short_strategy import LongShortStrategy
from simulator.strategy_simulator import strategy_simulator
from models.lstm_model import LSTMModel
from simulator.simulator_utils import get_config, get_logger
from ingestion.preprocess import load_tensors

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the trading simulation')
    parser.add_argument('--ckpt',
                        type=str,
                        default='checkpoints/last.ckpt',
                        help='Path to the model checkpoint')
    return parser.parse_args()

def main():
    # Get logger
    logger = get_logger()
    logger.info('Starting simulation')

    # Parse arguments
    args = parse_arguments()

    # Get configuration
    config = get_config()

    # Load test data and metadata
    data = load_tensors() # TODO: split data into train and test

    # Load last model checkpoint
    model = LSTMModel.load_from_checkpoint(args.ckpt)
    model.eval()

    # Create object of LongShortStrategy
    strategy = LongShortStrategy(
        config.get('LONG_COUNT', 10),
        config.get('SHORT_COUNT', 10)
    )

    # initialize data structures to store results
    ret_port = pd.Series(dtype=float)
    weights_db = pd.DataFrame(columns=['date', 'ticker', 'weights'])

    # loop through a range of time values
    simulation_days = config.get('SIMULATION_DAYS', 100)
    logger.info(f'Running simulation for {simulation_days} days')
    for t in range(1, simulation_days):
        forecast = model(data) # TODO: adjust to use number of days for sequence_length
        # use the strategy simulator to get portfolio's historical weights [weights_db]
        # and its next day returns [ret_port]
        ret_port, weights_db = strategy_simulator(
            path='results/',
            strategy=strategy,
            forecast=forecast,
            data=data,
            t=t,
            ret_port=ret_port,
            weights_db=weights_db
        )

    # Generate the performance report
    ret_port = pd.read_parquet('results/ret_port.parquet')
    ret_port['date'] = pd.to_datetime(ret_port['date'])
    ret_port.set_index('date', inplace=True)
    ret_port = ret_port['ret_port']
    qs.reports.html(ret_port, '^BVSP', text_description="""
    <p> Demonstration of a simple strategy</p>
    <p><strong>Important:</strong> Trading costs, taxes, and other fees were not considered in this simulation.</p>
    """)


if __name__ == '__main__':
    main()
