import torch
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import os

from simulator.strategy_interface import StrategyInterface


def strategy_simulator(
    path: str, 
    strategy: StrategyInterface,
    returns_ts: torch.Tensor, 
    forecast_ts: torch.Tensor,
    t: int,
    ret_port: pd.Series, 
    weights_db: pd.DataFrame, 
    metadata: dict,
    **kwargs
    ) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Calculate portfolio returns using the given strategy.

    Args:
        path (string): path to save strategy data
        strategy (StrategyInterface): Strategy according to the StrategyInterface
        returns_ts (torch.Tensor): returns for each stock.
        forecast_ts (torch.Tensor): predictions for each stock.
        t (int): Time value for calculation.
        ret_port (pd.Series): Accumulated portfolio returns.
        weights_db (pd.DataFrame): Accumulated weights database.

    Returns:
        pd.Series: Updated portfolio next day returns.
        pd.DataFrame: Updated weights database.
    """
    os.makedirs(path, exist_ok=True)

    # Calculate the weights for the specified t value
    weights = strategy.calculate_next_weights(forecast_ts, t=t, metadata=metadata, **kwargs) # weights.columns = ['ticker', 'weights', 'date', 'position']

    # Save a weights database
    weights_db = pd.concat([weights_db, weights], axis=0)
    weights_db.to_parquet(path + "weights_db.parquet")

    # Calculate the portfolio returns for the specified t value
    weights['returns'] = returns_ts[weights['idx'].values.astype(int)].numpy()
    weights.loc[weights['position'] == 'short', 'returns'] *= -1
    weights['returns'] *= weights['weights']
    weights['returns'] *= 0.01 # turn returns into float instead of percentage
    
    # Get date from metadata
    timestamp_idx = {v: k for k, v in metadata['timestamp_idx'].items()} # TODO: I'll have to commit metadata.pt to the repo
    date = timestamp_idx[t]
    ret_port[date] = weights['returns'].sum()

    # Save the portfolio returns
    aux = ret_port.reset_index()
    aux.columns = ['date', 'ret_port']
    aux.to_parquet(path + "ret_port.parquet")

    return ret_port, weights_db
