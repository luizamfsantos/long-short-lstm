from ingestion.get_data import pipeline, parse_arguments
import os

def test_pipeline():
    pipeline('2021-01-01', '2022-01-01', stock_list=['PETR4', 'VALE3'], config_path='config/credentials.yml', save_raw_data=True, path_to_save_raw_data='data/raw')
    assert os.path.exists('data/raw/PETR4_20210101_20220101.json'), 'Raw data not saved'

if __name__ == '__main__':
    test_pipeline()