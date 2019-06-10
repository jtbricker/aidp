"""This module defines classes with functionality for reading data from a file"""
from abc import ABC, abstractmethod
import logging
import pandas as pd

class DataFileReader(ABC):
    """Abstract Base Class for classes which can read in data files"""
    def __init__(self):
        self._logger = logging.getLogger(__name__)

    @abstractmethod
    def read_data(self, filepath):
        """Abstract method for reading a file and returning a dataframe

        Arguments:
            filepath {string} -- filepath of data file
        """

class ExcelDataReader(DataFileReader):
    """Defines logic for reading data from an excel sheet"""
    def read_data(self, filepath):
        """Reads data in from an excel sheet as a pandas DataFrame

        Arguments:
            filepath {string} -- filepath of the excel sheet
        """
        try:
            self._logger.info("Reading in input file: %s", filepath)
            return pd.read_excel(filepath)

        except FileNotFoundError:
            self._logger.exception('Error trying to read file: %s', filepath)
            raise