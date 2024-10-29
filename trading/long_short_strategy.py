from simulator.strategy_interface import StrategyInterface


class LongShortStrategy(StrategyInterface):
    def __init__(self):
        pass

    def calculate_next_weights(self, data, t, **kwargs):
        """
        Calculate the weights for the next period.

        Args:
            data (dict): Dictionary containing necessary data.
            t (int): Time value for calculation.

        Returns:
            pd.DataFrame: Weights for the next period.
        """
        #Get the data: model will receive forecast from LSTM
        prices = data['stocks'] # TODO: adjust to use data from raw_combined parquet files
        returns = prices.pct_change() # TODO: adjust to use column variacaopercent from data

        # Use LSTM to predict the direction of each stock
        # 

        # # Calculate the momentum
        # momentum = returns.rolling(window=252).mean().iloc[t - 1]

        # # Select the top and bottom stocks
        # top_stocks = momentum.head(10)
        # bottom_stocks = momentum.tail(10)

        # Calculate the weights: TODO: how to get the simulator know that bottom_stocks are short positions?
        weights = pd.concat([top_stocks, bottom_stocks], axis=0)
        weights = weights / weights.abs().sum()
        # return a dataframe with the columns date, ticker, and weights
        return weights.to_frame().T