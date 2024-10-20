import pandas as pd
import numpy as np
from ingestion.preprocess import (
    calculate_target_variable, 
    read_data)

def test_calculate_target_variables():
    df_test = pd.DataFrame({'returns': [-0.4, 0.2, -.3, .05, .01]})
    calculate_target_variable(df_test, 'returns')
    assert np.equal(df_test['target'].values.tolist(), [0, 1, 0, 1, 1]).all()


def test_read_data():
    data_path = 'data/raw_combined/2024/9'
    data_generator = read_data(data_path, batch_size=10)
    for batch in data_generator:
        data = batch.to_pandas()
        assert len(data.columns) > 0

if __name__ == '__main__':
    test_calculate_target_variables()
    test_read_data()