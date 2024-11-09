from typing import Dict, Protocol
import pandas as pd


class StrategyInterface(Protocol):
    def calculate_next_weights(self, data: Dict[str, pd.DataFrame], t: int) -> pd.DataFrame:
        """
        Calculate the weights for the next day or for the end of the day.

        Args:
            data (dict): Dictionary containing necessary data annotated with the keys, e.g., 'sp' and 'prices'.
            t (int): Iterator of time corresponding to today, according to your data dictionary.

        Returns:
            DataFrame: Weights for the next day or for the end of the day.
        """
        pass

    def check_return(self, weights: pd.DataFrame) -> bool:
        """
        Check if it is a valid return.
        """
        return all([c in weights.columns.values for c in ['date', 'ticker', 'weights', 'position']])
