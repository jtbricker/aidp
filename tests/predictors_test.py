"""Tests for the aidp.data.modeldata module"""
import unittest
from unittest.mock import Mock
import pandas as pd
import numpy as np
from aidp.ml.predictors import Predictor

class TestPredictor(unittest.TestCase):
    def test__load_model_from_file__sets_classifier(self):
        """ If the file exists, load the model """
        predictor = Predictor().load_model_from_file("both", "msa_v_pd_psp")

        assert hasattr(predictor, "classifier")

    def test__load_model_from_file__file_not_found__error(self):
        """ If the file doesn't exist, raise an error """
        with self.assertRaises(FileNotFoundError):
            Predictor().load_model_from_file("doesnt", "exist")

    def test__make_predictions__sets_predictions(self):
        """ Calls predict_proba method on model and sets the predictions variable """
        predictor = Predictor()
        predictor.classifier = Mock()
        mock_preds = np.array([[0.1, 0.9]])
        predictor.classifier.predict_proba = Mock(return_value=mock_preds)

        predictions = predictor.make_predictions(pd.DataFrame([{'GroupID':1}]))

        predictor.classifier.predict_proba.assert_called()
        assert mock_preds[:,1] == predictions, "Predictions should be set"