"""This module defines the different data experiments.

    Experiments are defined by what data is used when creating models.
    Some subset of the input data is used for each experiment.
 """
import logging
import pathlib
from abc import ABC, abstractmethod
import pandas as pd
from aidp.data.groupings import ParkinsonsVsControlGrouping, MsaPspVsPdGrouping, MsaVsPdPspGrouping, PspVsPdMsaGrouping, PspVsMsaGrouping
from aidp.ml.predictors import Predictor, LinearSvcPredictor

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
    trainer = LinearSvcPredictor()

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    @abstractmethod
    def filter_data(self, data):
        pass #pragma: no cover

    def predict(self, data):
        self._logger.info("Starting model prediction")
        filtered_data = self.filter_data(data)
        for grouping in self.groupings:
            self.predictor.load_model_from_file(self.key, grouping.key)
            grouping.predictions = self.predictor.make_predictions(filtered_data)
        self._logger.info("Starting model prediction")

    def train(self, data):
        self._logger.info("Starting model training")
        #TODO: Implement Training mechanism
        filtered_data = self.filter_data(data)
        for grouping in self.groupings:
            grouping.group_data(filtered_data).grouped_data
            self._logger.debug("Training model for grouping: %s", grouping.key)
            self.trainer.train_model(grouping.grouped_data) #Returns a Predictor() object
            # Write report of the results
            # Write model to pickle file 
        self._logger.debug("Finished model training")       

    def get_results(self):
        # TODO: Add tests
        results = pd.DataFrame()
        for grouping in self.groupings:
            column = '%s_%s (%s Probability)' %(self.key, grouping.key, grouping.positive_label)
            results[column] = grouping.predictions

        return results

    def __str__(self):
        return type(self).__name__

class ClinicalOnlyDataExperiment(DataExperiment):
    key = "clinical"

    def filter_data(self, data):
        # TODO: Add comment about what columns are being filtered
        standard_data = get_standardized_data(data)
        return standard_data[['GroupID', 'Age', 'Sex', 'UPDRS']]

class ImagingOnlyDataExperiment(DataExperiment):
    key = "dmri"

    def filter_data(self, data):
        # TODO: Add comment about what columns are being filtered
        standard_data = get_standardized_data(data)
        return standard_data.drop(['UPDRS'], axis=1)

class FullDataExperiment(DataExperiment):
    key = "both"

    def filter_data(self, data):
        # TODO: Add comment about what columns are being filtered
        return get_standardized_data(data)


def get_standardized_data(data):
    # TODO: Find a cleaner way to do this
    columns_conf = pathlib.Path(__file__).parent.parent.parent / 'resources/column_names.conf'
    with open(columns_conf) as f:
        columns = f.read().splitlines() 
        return data[columns]
