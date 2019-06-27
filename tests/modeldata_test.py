"""Tests for the aidp.data.modeldata module"""
import unittest
from unittest.mock import Mock
import pandas as pd
from aidp.data.modeldata import ModelData

class TestModelData(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mock_file_reader = Mock()
        self.model_data = ModelData("filename", self.mock_file_reader)

    def test__read_data__calls_data_reader_returns_self(self):
        result = self.model_data.read_data()

        self.mock_file_reader.read_data.assert_called()
        assert result == self.model_data

    def test__add_results__adds_columns_to_data(self):
        self.model_data.data = pd.DataFrame([{"A":1}, {"A":10}])
        new_data = pd.DataFrame([
            {"B":2, "C":3},
            {"B":20, "C":30}
        ])

        assert self.model_data.data.shape == (2,1)

        self.model_data.add_results(new_data)

        assert self.model_data.data.shape == (2,3)