import logging
import pandas as pd

class ModelData:
    def __init__(self, filename):
        self.logger = logging.getLogger(__name__)
        self.filename = filename
        self.data = None

    def read_data(self):
        """Reads in data as a pandas dataframe using the filename passed from the constructor
        """
        self.logger.info("Reading in input file: %s", self.filename)
        try:
            self.data = pd.read_excel(self.filename)
            return self

        except:
            self.logger.exception('Error trying to read file: %s', self.filename)
