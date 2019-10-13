""" This module defines logic around training machine learning models and using those model to make predictions """
import logging
import os
from datetime import datetime
import pickle
import pathlib

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
import pandas as pd
import numpy as np

import aidp.ml.helpers as ml

class Predictor():
    name = __name__
    _logger = logging.getLogger(__name__)

    def load_model_from_file(self, experiment_key, comparison_key, model_key="default"):
        filepath = pathlib.Path(__file__).parent.parent.parent / ('resources/models/%s/%s/%s.pkl' %(model_key, experiment_key, comparison_key))
        self._logger.info("Loading model from file: %s", filepath)
        
        try:
            with open(str(filepath), "rb") as f:
                self.classifier = pickle.load(f)
            
            self._logger.debug("Sucessfully loaded model: %s", self.classifier)
            assert isinstance(self.classifier, BaseEstimator), "Loaded model should be an estimator" 

        except FileNotFoundError:
            self._logger.exception('Model file not found: %s.  Was this file moved?', filepath)
            raise
        
        self._logger.debug("Finished loading model from file") 
        return self
        
    def save_model_to_file(self, experiment_key, comparison_key, model_key):        
        try:
            filepath = pathlib.Path(__file__).parent.parent.parent / ('resources/models/%s/%s/%s.pkl' %(model_key, experiment_key, comparison_key))
        
            #Make sure the directory exists
            os.makedirs(os.path.dirname(str(filepath)), exist_ok=True)

            self._logger.info("Attempting to save model to file: %s", filepath)        
            with open(str(filepath), 'wb') as pickle_file:
                pickle.dump(self.classifier, pickle_file)

        except:
            self._logger.exception('Failed to save the model.  Attempting to continue')
            self._logger.warn("Model not saved for experiment: %s and comaparison: %s" %(experiment_key, comparison_key))
            raise
        
        self._logger.debug("Sucessfully saved model: %s", self.classifier)
        return self
    
    def make_predictions(self, data):
        self._logger.info("Making predictions")
        # We don't consider the actual GroupID when making predictions
        prediction_data = data.drop(['GroupID'], axis=1)
        predictions = self.classifier.predict_proba(prediction_data)
        return predictions[:,1]

class LinearSvcPredictor(Predictor):
    name = "Linear SVC Model"
    def __init__(self):
        self.param_grid = {
            "classifier__C": np.logspace(-5, 2, 20),
        },

        self.classifier = Pipeline([
            ('Scaler', StandardScaler()),
            ('classifier', SVC(kernel='linear', class_weight='balanced', probability=True))
        ])

        self.cv=5
        self.test_size=0.20
        self.scoring_list={
            'recall':make_scorer(recall_score),
            'precision':make_scorer(precision_score),
            'auc':make_scorer(roc_auc_score),
            'specificity':make_scorer(ml.specificity),
            'npv':make_scorer(ml.negative_predictive_value),
            'accuracy':make_scorer(accuracy_score),
            'weighted_sensitivity':make_scorer(ml.weighted_sensitivity),
            'weighted_ppv':make_scorer(ml.weighted_ppv),
            'weighted_specificity':make_scorer(ml.weighted_specificity),
            'weighted_npv':make_scorer(ml.weighted_npv),
            'weighted_accuracy':make_scorer(ml.weighted_accuracy)
        }

        self.scoring='f1_micro'
        self.random_seed = 55

    def train_model(self, data):
        self._logger.info("\tTraining %s Model", self.name)
        y = data['GroupID']
        X = data.drop(['GroupID'], axis=1)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_seed)
        self.classifier = ml.grid_search_optimization(self.classifier, self.param_grid, X_train, Y_train, X_test, Y_test, cv= self.cv, scoring= self.scoring)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        