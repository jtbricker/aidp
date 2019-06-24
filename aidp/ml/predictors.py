""" This module defines code around making predictions using machine learning models """
import logging
import pickle
from sklearn.base import BaseEstimator

class Predictor():
    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def load_model_from_file(self, experiment_key, comparison_key):
        filepath = "./resources/models/%s/%s.pkl" %(experiment_key, comparison_key)
        self._logger.info("Loading model from file: %s", filepath)
        
        try:
            with open(filepath, "rb") as f:
                self.prediction_model = pickle.load(f)
            
            self._logger.debug("Sucessfully loaded model: %s", self.prediction_model)
            assert isinstance(self.prediction_model, BaseEstimator), "Loaded model should be an estimator" 

        except FileNotFoundError:
            self._logger.exception('Model file not found: %s.  Was this file moved?', filepath)
            raise
        
        self._logger.debug("Finished loading model from file") 
        return self
        
    def make_predictions(self, data):
        self._logger.info("Making predictions")
        return self.prediction_model.predict_proba(data.drop(['GroupID'], axis=1))[:,1]