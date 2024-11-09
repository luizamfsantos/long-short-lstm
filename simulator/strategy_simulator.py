import torch
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import os

from simulator.strategy_interface import StrategyInterface


def strategy_simulator(
    path: str, 
    strategy: StrategyInterface,
    return_list: list[torch.Tensor], 
    forecast: list[torch.Tensor],
    t: int,
    ret_port: pd.Series, 
    weights_db: pd.DataFrame, 
    **kwargs
    ) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Calculate portfolio returns using the given strategy.

    Args:
        path (string): path to save strategy data
        strategy (StrategyInterface): Strategy according to the StrategyInterface
        return_list (list[torch.Tensor]): list of tensors with returns for each stock.
        forecast (list[torch.Tensor]): predictions for each stock.
        t (int): Time value for calculation.
        ret_port (pd.Series): Accumulated portfolio returns.
        weights_db (pd.DataFrame): Accumulated weights database.

    Returns:
        pd.Series: Updated portfolio next day returns.
        pd.DataFrame: Updated weights database.
    """
    os.makedirs(path, exist_ok=True)

    # Calculate the weights for the specified t value
    weights = strategy.calculate_next_weights(forecast, t=t) # weights.columns = ['ticker', 'weights', 'date', 'position']

    # Save a weights database
    weights_db = pd.concat([weights_db, weights], axis=0)
    weights_db.to_parquet(path + "weights_db.parquet")

    # Calculate and save portfolio returns TODO: adjust to use column variacaopercent from data
    prices = data['stocks']
    prices_1 = prices[weights.ticker].loc[prices.index[t - 1:t + 1]] 
    returns_1 = np.log(prices_1).diff().tail(1).mean() # TODO: modify this to allow for short positions
    weights_index = weights.weights
    weights_index.index = weights.ticker
    ret_port[prices.index[t]] = returns_1 @ weights_index

    # Save the portfolio returns
    aux = ret_port.reset_index()
    aux.columns = ['date', 'ret_port']
    aux.to_parquet(path + "ret_port.parquet")

    return ret_port, weights_db
