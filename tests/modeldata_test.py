"""Tests for the data.modeldata module"""
import unittest
from unittest.mock import Mock
import aidp.data.modeldata as modeldata

class TestModelData(unittest.TestCase):
    def test_read_data_calls_data_reader_returns_self(self):
        mock_file_reader = Mock()
        model_data = modeldata.ModelData("filename", mock_file_reader)

        result = model_data.read_data()

        mock_file_reader.read_data.assert_called()
        assert result == model_data