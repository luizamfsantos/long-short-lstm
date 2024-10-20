import pandas as pd
import numpy as np
from typing import Dict, Tuple
import os

from simulator.strategy_interface import StrategyInterface


def strategy_simulator(
    path: str, 
    strategy: StrategyInterface,
    data: Dict[str, pd.DataFrame], 
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
        data (dict): Dictionary containing necessary data.
        t (int): Time value for calculation.
        ret_port (pd.Series): Accumulated portfolio returns.
        weights_db (pd.DataFrame): Accumulated weights database.

    Returns:
        pd.Series: Updated portfolio next day returns.
        pd.DataFrame: Updated weights database.
    """

    # If path does not exist, create it
    if not os.path.exists(path):
        os.makedirs(path)

    # Calculate the weights for the specified t value
    weights = strategy.calculate_next_weights(data, t=t, **kwargs)

    # Save a weights database
    weights_db = pd.concat([weights_db, weights], axis=0)
    weights_db.to_parquet(path + "weights_db.parquet")

    # Calculate and save portfolio returns
    prices = data['stocks']
    prices_1 = prices[weights.ticker].loc[prices.index[t - 1:t + 1]]
    returns_1 = np.log(prices_1).diff().tail(1).mean()
    weights_index = weights.weights
    weights_index.index = weights.ticker
    ret_port[prices.index[t]] = returns_1 @ weights_index

    aux = ret_port.reset_index()
    aux.columns = ['date', 'ret_port']
    aux.to_parquet(path + "ret_port.parquet")

    return ret_port, weights_db
