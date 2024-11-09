import argparse
import pandas as pd
import quantstats as qs
import torch
from trading.long_short_strategy import LongShortStrategy
from simulator.strategy_simulator import strategy_simulator
from models.lstm_model import LSTMModel
from simulator.simulator_utils import get_config, get_logger
from models.data_preparation import TimeSeriesDataModule

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

    # Load data 
    data_module = TimeSeriesDataModule(
        batch_size=1,
        sequence_length=config.get('SEQUENCE_LENGTH', 5),
    )

    # Load last model checkpoint
    model = LSTMModel.load_from_checkpoint(args.ckpt).to('cpu')
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
    
    # Prepare the dataloader for the test dataset
    data_module.setup()
    test_dataloader = data_module.test_dataloader()

    # Run inference on the test dataset
    # TODO: make this automatic later, but for now in the input_sequence,
    # the 2nd value in the feature vector is the return of the previous day
    return_list = []
    all_predictions = [] # idx: timestamp, [num_stocks] tensor float (0, 1)
    with torch.no_grad():
        # num_batch = 1
        for batch in test_dataloader:
            input_sequence, _ = batch # [1, num_tickers, seq_len, num_features]
            return_list.append(input_sequence[:, :, -1, 1].squeeze()) # [1, num_tickers]
            prediction = model(input_sequence) # [1, num_tickers, 1]
            prediction = prediction.squeeze() # [num_tickers]
            all_predictions.append(prediction)
    return_list.pop(0) # remove the first day so it represents the returns of the day it predicts
    number_of_predictions = len(all_predictions)
    assert number_of_predictions == simulation_days - 2, 'Simulation days is longer than the test dataset'
    for t in range(number_of_predictions - simulation_days - 2, number_of_predictions - 2):
        # use the strategy simulator to get portfolio's historical weights [weights_db]
        # and its next day returns [ret_port]
        ret_port, weights_db = strategy_simulator(
            path='results/',
            strategy=strategy,
            forecast=all_predictions[t],
            return_list=return_list[t],
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
