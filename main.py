import pandas as pd
import quantstats as qs
from trading.long_short_strategy import LongShortStrategy
from simulator.strategy_simulator import strategy_simulator
from models.lstm_model import LSTMModel

def main():
    # TODO: load test data
    data = []

    # TODO: load last model checkpoint
    model = LSTMModel.load_from_checkpoint('checkpoints/lstm_model.ckpt')
    model.eval()
    forecast = model.predict(data)

    # TODO: create object of LongShortStrategy
    strategy = LongShortStrategy()
    # check execution for one day
    weights = strategy.calculate_next_weights(data, t=1)
    assert stategy.check_return(weights), 'Could not calculate weights or invalid return'

    # initialize data structures to store results
    ret_port = pd.Series(dtype=float)
    weights_db = pd.DataFrame(columns=['date', 'ticker', 'weights'])

    # TODO: add argsparse to get the number of days to run the simulation
    # loop through a range of time values
    for t in range(1, 10):
        # use the strategy simulator to get portfolio's historical weights [weights_db]
        # and its next day returns [ret_port]
        ret_port, weights_db = strategy_simulator(
            path='results/',
            strategy=strategy,
            data=data,
            t=t,
            ret_port=ret_port,
            weights_db=weights_db
        )
    
    # Generate the performance report
    ret_port = pd.read_parquet('results/ret_port.parquet')
    ret_port['date'] = pd.to_datetime(ret_port['date'])
    ret_port.set_index('date', inplace=True)
    ret_port = ret_port['ret_port'] # get only the returns column
    qs.reports.html(ret_port,'^BVSP', text_description="""
    <p> Demonstration of a simple strategy</p>
    <p><strong>Important:</strong> Trading costs, taxes, and other fees were not considered in this simulation.</p>
    """)


if __name__ == '__main__':
    main()