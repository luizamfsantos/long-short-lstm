import pandas as pd
import numpy as np
from ingestion.preprocess import (
    calculate_target_variable, 
    calculate_returns,
    read_data)

def test_calculate_target_variables():
    df_test = pd.DataFrame({'price': [1, 2, 2, 4, 5]})
    calculate_target_variable(df_test, 'price')
    assert np.equal(df_test['target'].values.tolist(), [False, True, False, True, True]).all()

def test_calculate_returns():
    df_test = pd.DataFrame({'price': [1, 2, 2, 4, 5]})
    calculate_returns(df_test, 'price')
    assert np.isclose(df_test['returns'].values.tolist(), [np.nan, 1, 0, 1, 0.25], equal_nan=True).all()

def test_read_data():
    data_path = 'data/raw_combined/2024/9'
    data_generator = read_data(data_path, batch_size=10)
    for batch in data_generator:
        data = batch.to_pandas()
        assert len(data.columns) > 0

if __name__ == '__main__':
    test_calculate_target_variables()
    test_calculate_returns()
    test_read_data()