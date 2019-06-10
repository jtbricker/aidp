"""Tests for the data.reader module"""
import unittest
import pandas as pd
import aidp.data.reader as reader

class TestExcelDataReader(unittest.TestCase):
    def test_read_data_filenotfound_raisesexception(self):
        excel_reader = reader.ExcelDataReader()

        with self.assertRaises(FileNotFoundError):
            excel_reader.read_data("thisFileDoesntExist.xlsx")

    def test_read_data_filefound_returndataframe(self):
        excel_reader = reader.ExcelDataReader()

        result = excel_reader.read_data("./tests/resources/test.xlsx")

        assert isinstance(result, pd.DataFrame)
