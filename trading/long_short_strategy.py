from simulator.strategy_interface import StrategyInterface
import pandas as pd


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
        # Ensure that long_count and short_count are smaller than the number of stocks
        forecast_length = forecast.shape[0]
        if self.long_count > forecast_length or self.short_count > forecast_length:
            long_count = forecast_length // 2  # top half
            short_count = forecast_length - long_count  # bottom half
        else:
            long_count = self.long_count
            short_count = self.short_count
        # Select the top and bottom stocks
        top_stocks_idx = forecast.topk(long_count).indices
        top_stocks_names = ...  # TODO: get the names of the top stocks
        bottom_stocks_idx = (-1*forecast).topk(short_count).indices
        bottom_stocks_names = ...  # TODO: get the names of the top stocks
        top_stocks = pd.DataFrame(
            {'ticker': top_stocks_names, 'weights': 1, 'date': t, 'position': 'long'})
        bottom_stocks = pd.DataFrame(
            {'ticker': bottom_stocks_names, 'weights': 1, 'date': t, 'position': 'short'})
        weights_df = pd.concat([top_stocks, bottom_stocks], axis=0)
        weights_df['weights'] = weights_df['weights'] / \
            weights_df['weights'].abs().sum()  # Normalize the weights
        # return a dataframe with the columns date, ticker, weights and position
        return weights.to_frame().T
