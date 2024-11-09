from simulator.strategy_interface import StrategyInterface
import pandas as pd
import numpy as np


class LongShortStrategy(StrategyInterface):
    def __init__(self,
                 long_count: int = 10,
                 short_count: int = 10):
        self.long_count = long_count
        self.short_count = short_count

    def calculate_next_weights(
            self,
            forecast,
            metadata,
            t,
            threshold: float = 0.01,
            **kwargs):
        """
        Calculate the weights for the next period.

        Args:
            forecast (torch.Tensor): predictions for each stock. (num_tickers, 1)
                # Strategy will receive forecast from LSTM 
                # near to 1 means buy, near to 0 means sell
            metadata : information to retrieve the stock names 
            t (int): Time value for calculation.

        Returns:
            pd.DataFrame: Weights for the next period.
        """
        forecast = forecast.numpy()
        # uncertainty = 0.5
        # long_predictions = forecast > 0.5 + threshold
        long_predictions = (forecast > 0.5 + threshold).sum()
        # short_predictions = forecast < 0.5 - threshold
        short_predictions = (forecast < 0.5 - threshold).sum()
        # Ensure that long_count and short_count 
        # are smaller than the number of stocks predicted
        # to be long and short
        long_count = min(self.long_count, long_predictions)
        short_count = min(self.short_count, short_predictions)
        
        ticker_idx = {v: k for k, v in metadata['ticker_idx'].items()} # TODO: I'll have to commit metadata.pt to the repo
        # Select the top and bottom stocks
        # TODO: idea modify weights so that it uses the confidence of the prediction
        # to determine the weight of the stock
        if long_count > 0:
            top_stocks_idx = np.argsort(forecast, axis=0)[-long_count:]
            top_stocks_names = ticker_idx[top_stocks_idx]
            top_stocks = pd.DataFrame(
            {'ticker': top_stocks_names, 'weights': 1, 'date': t, 'position': 'long', 'idx': top_stocks_idx})
        else:
            top_stocks = pd.DataFrame(columns=['ticker', 'weights', 'date', 'position', 'idx'])
        if short_count > 0:
            bottom_stocks_idx = np.argsort(forecast, axis=0)[:short_count]
            bottom_stocks_names = ticker_idx[bottom_stocks_idx]
            bottom_stocks = pd.DataFrame(
                {'ticker': bottom_stocks_names, 'weights': 1, 'date': t, 'position': 'short', 'idx': bottom_stocks_idx})
        else:
            bottom_stocks = pd.DataFrame(columns=['ticker', 'weights', 'date', 'position', 'idx'])
        weights_df = pd.concat([top_stocks, bottom_stocks], axis=0).reset_index(drop=True)
        
        # Normalize the weights
        weights_df['weights'] = weights_df['weights'] / \
            weights_df['weights'].abs().sum()  
        
        # return a dataframe with the columns date, ticker, weights and position
        return weights_df
