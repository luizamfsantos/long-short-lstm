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
        # Get the data
        prices = data['stocks']
        returns = prices.pct_change()

        # Calculate the momentum
        momentum = returns.rolling(window=252).mean().iloc[t - 1]

        # Sort the momentum
        momentum = momentum.sort_values(ascending=False)

        # Select the top and bottom stocks
        top_stocks = momentum.head(10)
        bottom_stocks = momentum.tail(10)

        # Calculate the weights
        weights = pd.concat([top_stocks, bottom_stocks], axis=0)
        weights = weights / weights.abs().sum()
        # return a dataframe with the columns date, ticker, and weights
        return weights.to_frame().T