"""This module defines the different data experiments.

    Experiments are defined by what data is used when creating models.
    Some subset of the input data is used for each experiment.
 """
import logging
from abc import ABC, abstractmethod
from aidp.data.groupings import ParkinsonsVsControlGrouping, MsaPspVsPdGrouping, MsaVsPdPspGrouping, PspVsPdMsaGrouping, PspVsMsaGrouping
from aidp.ml.predictors import Predictor

class DataExperiment(ABC):
    key = None
    groupings = [
        ParkinsonsVsControlGrouping(),
        MsaPspVsPdGrouping(),
        MsaVsPdPspGrouping(),
        PspVsPdMsaGrouping(),
        PspVsMsaGrouping()
    ]
    predictor = Predictor()

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    @abstractmethod
    def filter_data(self, data):
        pass #pragma: no cover

    def predict(self, data):
        self._logger.info("Starting model prediction")
        self.filtered_data = self.filter_data(data)
        for grouping in self.groupings:
            self.predictor.load_model_from_file(self.key, grouping.key)
            grouping.predictions = self.predictor.make_predictions(self.filtered_data)
        self._logger.info("Starting model prediction")

    def train(self):
        self._logger.info("Starting model training")
        #TODO: Implement Training mechanism
        self._logger.debug("Finished model training")       

    def __str__(self):
        return type(self).__name__

class ClinicalOnlyDataExperiment(DataExperiment):
    key = "clinical"

    def filter_data(self, data):
        standard_data = get_standardized_data(data)
        return standard_data[['GroupID', 'Age', 'Sex', 'UPDRS']]

class ImagingOnlyDataExperiment(DataExperiment):
    key = "dmri"

    def filter_data(self, data):
        standard_data = get_standardized_data(data)
        return standard_data.drop(['UPDRS'], axis=1)

class FullDataExperiment(DataExperiment):
    key = "both"

    def filter_data(self, data):
        return get_standardized_data(data)


def get_standardized_data(data):
    with open('./resources/column_names.conf') as f:
        columns = f.read().splitlines() 
        return data[columns]
