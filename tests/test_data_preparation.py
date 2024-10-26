from models.data_preparation import TimeSeriesData
import pytest
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from unittest.mock import patch, MagicMock


class TestTimeSeriesData:
    @pytest.fixture
    def mock_generator(self):
        """ Create a mock generator that yields tensors """
        def create_batch(num_tickers, num_timestamps, num_features):
            X = torch.rand(num_tickers, num_timestamps, num_features)
            y = torch.rand(num_tickers, num_timestamps, 1)
            metadata = {'num_batches': 1}
            return X, y, metadata

        batch1 = create_batch(5, 100, 4)
        batch2 = create_batch(5, 100, 4)
        return [batch1, batch2]

    @pytest.fixture
    def dataset(self, mock_generator):
        """ Create a dataset instance with mocked generator """
        with patch('models.data_preparation.load_tensors') as mock_load_tensors:
            mock_load_tensors.return_value = iter(mock_generator)
            return TimeSeriesData(
                seq_len=2,
                tensor_path='dummy/path',
                target_path='dummy/path',
                metadata_path='dummy/path'
            )

    def test_init(self, dataset, mock_generator):
        """ Test if the dataset is initialized correctly """
        assert dataset.seq_len == 2
        assert dataset.tensor_path == 'dummy/path'
        assert dataset.target_path == 'dummy/path'
        assert dataset.metadata_path == 'dummy/path'
        assert dataset.current_X.shape == mock_generator[0][0].shape
        assert dataset.current_y.shape == mock_generator[0][1].shape
        assert dataset.total_length == 100
        assert dataset.batch_length == 100
        assert dataset.batch_start_idx == 0

    # def test_len(self, dataset):
    #     """ Test if __len__ returns correct total legth"""
    #     assert len(dataset) == 6

    def test_getitem_within_batch(self, dataset):
        """ Test if __getitem__ returns correct sequence within the current batch"""
        idx = 4
        X, y = dataset[idx]

        assert X.shape == (5, 2, 4)  # 5 tickers, 2 timestamps, 4 features
        assert y.shape == (5, 2, 1)  # 5 tickers, 2 timestamps, 1 target

        # Check if the returned sequence is correct
        assert torch.equal(X, dataset.current_X[:, idx:idx+2, :])
        assert torch.equal(y, dataset.current_y[:, idx:idx+2, :])

    def test_getitem_across_batches(self, dataset):
        """ Test if __getitem__ returns correct sequence across two batches"""
        idx = 99
        X, y = dataset[idx]
        logging.debug(f"X: {X.shape}")
        assert X.shape == (5, 2, 4)
        assert y.shape == (5, 2, 1)

        # TODO: Check if the returned sequence is correct
        # problem to test, the current_X and current_y don't have the
        # previous batch data, so torch.equal can't be used

    def test_getitem_exactly_at_the_end_of_batch(self, dataset):
        """ Test if __getitem__ returns correct sequence at the end of the batch"""
        idx = 100
        X, y = dataset[idx]

        assert X.shape == (5, 2, 4)
        assert y.shape == (5, 2, 1)

        # Check if the returned sequence is correct
        assert torch.equal(X, dataset.current_X[:, 0:2, :])
        assert torch.equal(y, dataset.current_y[:, 0:2, :])

    def test_getitem_out_of_bounds(self, dataset):
        """ Test if __getitem__ raises IndexError when the index is out of bounds """
        with pytest.raises(IndexError):
            dataset[201] # should raise IndexError after exhausting all the data

    def test_with_dataloader(self, dataset):
        """ Test integration with DataLoader """
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False
            )
        for i, (X, y) in enumerate(dataloader):
            batch_size, num_tickers, seq_len, num_features = X.size()
            assert batch_size == 2
            assert num_tickers == 5
            assert seq_len == 2
            assert num_features == 4
            assert y.shape == (2, 5, 2, 1) # batch_size, num_tickers, seq_len, 1