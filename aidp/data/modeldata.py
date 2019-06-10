"""This module contains logic related the input and output files of the aidp model"""
import pandas as pd

class ModelData:
    """This class represents the data and includes """
    def __init__(self, filename, dataReader):
        self._data_reader = dataReader
        self.filename = filename
        self.data = None

    def read_data(self):
        """Reads in data as a pandas dataframe using the filename passed from the constructor
        """
        self._data_reader.read_data(self.filename)

        return self 
