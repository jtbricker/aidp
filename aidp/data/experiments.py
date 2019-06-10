"""This module defines the different data experiments.

    Experiments are defined by what data is used when creating models.
    Some subset of the input data is used for each experiment.
 """
from abc import ABC, abstractmethod

class DataExperiment(ABC):
    def __init__(self, model_data):
        self.filtered_data = self.filter_data(model_data)

    @abstractmethod
    def filter_data(self, model_data):
        pass

    def predict(self):
        #TODO: Implement Prediction mechanism
        pass

    def train(self):
        #TODO: Implement Training mechanism
        pass

class ClinicalOnlyDataExperiment(DataExperiment):
    def filter_data(self, model_data):
        #TODO: Define how to filter data for experiment
        pass

class ImagingOnlyDataExperiment(DataExperiment):
    def filter_data(self, model_data):
        #TODO: Define how to filter data for experiment
        pass

class FullDataExperiment(DataExperiment):
    def filter_data(self, model_data):
        #TODO: Define how to filter data for experiment
        pass
