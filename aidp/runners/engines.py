"""This module defines execution engines that will perform work"""
from abc import ABC, abstractmethod
import logging
from aidp.data.experiments import ClinicalOnlyDataExperiment, ImagingOnlyDataExperiment, \
     FullDataExperiment

class Engine(ABC):
    """Abstract Base Class for classes which execute a series of related tasks"""
    def __init__(self):
        self._logger = logging.getLogger(__name__)

    @abstractmethod
    def start(self):
        """Abstract method for executing the engine's tasks"""

class PredictionEngine(Engine):
    """Defines tasks that will be completed as part of the prediction workflow"""
    def __init__(self, model_data):
        self.experiments = [
            FullDataExperiment(model_data),
            ImagingOnlyDataExperiment(model_data),
            ClinicalOnlyDataExperiment(model_data)
        ]
        super(PredictionEngine, self).__init__()

    def start(self):
        for experiment in self.experiments:
            #TODO: add logging0
            experiment.predict()
            # TODO: Do something with the results of the prediction

#TODO: Define a TrainingEngine class
