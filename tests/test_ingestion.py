import pytest
import os
import json
from datetime import datetime 
from ingestion.get_data import pipeline, extract_data_from_api_response, combine_data

@pytest.fixture
def sample_market_data_list():
    return [
        {
            'tab0': {
                'vol_mm_reais': {
                    '2024-10-01T00:00:00': '0.245092',
                    '2024-10-02T00:00:00': '0.442891',
                    '2024-10-03T00:00:00': '0.35257',
                    '2024-10-04T00:00:00': '0.314292',
                    '2024-10-07T00:00:00': '0.264526',
                    '2024-10-08T00:00:00': '1.024414',
                    '2024-10-09T00:00:00': '0.401199',
                    '2024-10-10T00:00:00': '0.214889',
                    '2024-10-11T00:00:00': '0.179387'
                }
            },
            'ticker': 'PETR4'
        },
        {
            'tab0': {
                'vol_mm_reais': {
                    '2024-10-01T00:00:00': '0.245092',
                    '2024-10-02T00:00:00': '0.442891',
                    '2024-10-03T00:00:00': '0.35257',
                    '2024-10-04T00:00:00': '0.314292',
                    '2024-10-07T00:00:00': '0.264526',
                    '2024-10-08T00:00:00': '1.024414',
                    '2024-10-09T00:00:00': '0.401199',
                    '2024-10-10T00:00:00': '0.214889',
                    '2024-10-11T00:00:00': '0.179387'
                }
            },
            'ticker': 'VALE3'
        },
        {
            'tab0': {
                'vol_mm_reais': {}
            },
            'ticker': 'PETR3'
        }
    ]

@pytest.fixture
def sample_extracted_data():
    return [
        {
            'vol_mm_reais': {
                '2024-10-01T00:00:00': '0.245092',
                '2024-10-02T00:00:00': '0.442891',
                '2024-10-03T00:00:00': '0.35257',
                '2024-10-04T00:00:00': '0.314292',
                '2024-10-07T00:00:00': '0.264526',
                '2024-10-08T00:00:00': '1.024414',
                '2024-10-09T00:00:00': '0.401199',
                '2024-10-10T00:00:00': '0.214889',
                '2024-10-11T00:00:00': '0.179387'
            },
            'ticker': {
                '2024-10-01T00:00:00': 'PETR4',
                '2024-10-02T00:00:00': 'PETR4',
                '2024-10-03T00:00:00': 'PETR4',
                '2024-10-04T00:00:00': 'PETR4',
                '2024-10-07T00:00:00': 'PETR4',
                '2024-10-08T00:00:00': 'PETR4',
                '2024-10-09T00:00:00': 'PETR4',
                '2024-10-10T00:00:00': 'PETR4',
                '2024-10-11T00:00:00': 'PETR4'
            }
        },
        {
            'vol_mm_reais': {
                '2024-10-01T00:00:00': '0.245092',
                '2024-10-02T00:00:00': '0.442891',
                '2024-10-03T00:00:00': '0.35257',
                '2024-10-04T00:00:00': '0.314292',
                '2024-10-07T00:00:00': '0.264526',
                '2024-10-08T00:00:00': '1.024414',
                '2024-10-09T00:00:00': '0.401199',
                '2024-10-10T00:00:00': '0.214889',
                '2024-10-11T00:00:00': '0.179387'
            },
            'ticker': {
                '2024-10-01T00:00:00': 'VALE3',
                '2024-10-02T00:00:00': 'VALE3',
                '2024-10-03T00:00:00': 'VALE3',
                '2024-10-04T00:00:00': 'VALE3',
                '2024-10-07T00:00:00': 'VALE3',
                '2024-10-08T00:00:00': 'VALE3',
                '2024-10-09T00:00:00': 'VALE3',
                '2024-10-10T00:00:00': 'VALE3',
                '2024-10-11T00:00:00': 'VALE3'
            }
        }
    ]

@pytest.mark.integration
def test_save_raw_data(tmp_path):
    """ Test that raw data is saved correctly """
    # Create a temporary directory for the test
    path_to_save_raw_data = tmp_path / 'raw'
    path_to_save_raw_data.mkdir()

    # Run the pipeline with save_raw_data=True
    pipeline(
        '2021-01-01',
        '2022-01-01',
        stock_list=['PETR4', 'VALE3'],
        config_path='config/credentials.yml',
        save_raw_data=True,
        path_to_save_raw_data=str(path_to_save_raw_data)
    )

    # Check that the raw data was saved correctly
    expected_files = [
        'PETR4_20210101_20220101.json',
        'VALE3_20210101_20220101.json'
    ]

    for file in expected_files:
        assert (path_to_save_raw_data / file).exists()

        # Clean up the temporary directory
        (path_to_save_raw_data / file).unlink()

def test_extract_data_from_api_response(sample_market_data_list, sample_extracted_data):
    """ Test that data is extracted correctly from the API response """
    extracted_data = extract_data_from_api_response(sample_market_data_list)
    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 2 # should combine VEL3 and PETR4 and ignore empty data for PETR3
    assert 'vol_mm_reais' in extracted_data[0]
    assert 'ticker' in extracted_data[0]
    assert extracted_data == sample_extracted_data
    assert all(value == 'PETR4' for value in extracted_data[0]['ticker'].values())

    # Check first entry 
    assert extracted_data[0]['vol_mm_reais']['2024-10-01T00:00:00'] == '0.245092'

def test_combine_data(sample_extracted_data):
    """ Test that data is correctly combined into a single DataFrame """
    combined_data = combine_data(sample_extracted_data)
    assert not combined_data.empty
    assert 'vol_mm_reais' in combined_data.columns
    assert 'ticker' in combined_data.columns
    assert len(combined_data) == 18 # 9 days for each of the 2 stocks
    assert combined_data['ticker'].nunique() == 2
    assert set(combined_data['ticker'].unique()) == {'PETR4', 'VALE3'}
    #assert combined_data['vol_mm_reais'].dtype == 'float64' # TODO: fix data type