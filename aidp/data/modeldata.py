"""This module contains logic related the input and output files of the aidp model"""
import os
import logging
import pandas as pd

class ModelData:
    """This class represents the data and includes """
    _logger =logging.getLogger(__name__)
    
    def __init__(self, filename, dataReader):
        self._data_reader = dataReader
        self.filename = filename
        self.data = None

    def read_data(self):
        """Reads in data as a pandas dataframe using the filename passed from the constructor"""
        self.data = self._data_reader.read_data(self.filename)

        return self 

    def add_results(self, new_data):
        """Adds additional columns (predictions) to the class' data DataFrame"""
        #TODO: Add tests
        self._logger.debug("Adding %s to the data", ", ".join(new_data.columns))
        self.data = self.data.join(new_data) 

    def write_output_file(self):
        """Writes the classes data DataFrame to disk as an excel sheet"""
        #TODO: Add Tests
        #FUTURE: Extract DataWriter class
        new_filename = '%s_out.xlsx' %os.path.splitext(os.path.basename(self.filename))[-2]
        self._logger.info("Writing results to output file: %s", new_filename)
        self.data.to_excel(new_filename)
