import pandas as pd
import numpy as np
from ingestion.preprocess import calculate_target_variable, calculate_returns

def test_calculate_target_variables():
    df_test = pd.DataFrame({'price': [1, 2, 2, 4, 5]})
    calculate_target_variable(df_test, 'price')
    assert np.equal(df_test['target'].values.tolist(), [False, True, False, True, True]).all()

def test_calculate_returns():
    df_test = pd.DataFrame({'price': [1, 2, 2, 4, 5]})
    calculate_returns(df_test, 'price')
    print(df_test)
    assert np.isclose(df_test['returns'].values.tolist(), [np.nan, 1, 0, 1, 0.25], equal_nan=True).all()
if __name__ == '__main__':
    test_calculate_target_variables()
    test_calculate_returns()