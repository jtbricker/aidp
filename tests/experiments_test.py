"""Test for the aidp.runners.experiments module """
import unittest
from unittest.mock import patch, Mock
from aidp.data.experiments import get_standardized_data, DataExperiment, ClinicalOnlyDataExperiment, ImagingOnlyDataExperiment, FullDataExperiment
from aidp.data.modeldata import ModelData
from aidp.data.reader import ExcelDataReader
import pandas as pd

class TestExperiments(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_data = ModelData('./tests/resources/test.xlsx', ExcelDataReader()).read_data()

    def test__get_standardized_data__columns_not_in_data__throws_exception(self):
        data = pd.DataFrame()
    
        with self.assertRaises(KeyError):
            get_standardized_data(data)

    def test__get_standardized_data__columns_in_data__(self):    
        get_standardized_data(self.model_data.data)

    def test__ClinicalOnlyDataExperiment__filter_data__correct_columns(self):
        """ Filters to the correct data for the ClinicalOnlyDataExperiment """
        filtered_data = ClinicalOnlyDataExperiment().filter_data(self.model_data.data)

        assert len(filtered_data.columns) == 4
        for col in ['GroupID', 'Age', 'Sex', 'UPDRS']:
            assert col in filtered_data.columns

    def test__ImagingOnlyDataExperiment__filter_data__correct_columns(self):
        """ Filters to the correct data for the ImagingOnlyDataExperiment """
        filtered_data = ImagingOnlyDataExperiment().filter_data(self.model_data.data)

        assert len(filtered_data.columns) == 123
        assert 'UPDRS' not in filtered_data.columns        
        

    def test__FullDataExperiment__filter_data__correct_columns(self):
        """ Filters to the correct data for the FullDataExperiment """
        filtered_data = FullDataExperiment().filter_data(self.model_data.data)

        assert len(filtered_data.columns) == 124

    def test__DataExperiment_predict__executes_prediction_workflow(self):
        data = pd.DataFrame()
        mock_grouping = Mock()
        mock_predictor = Mock()

        experiment = DataExperimentTest()
        experiment.groupings = [mock_grouping, Mock(), Mock()]
        experiment.predictor = mock_predictor


        with patch.object(experiment, 'filter_data') as mock_filter_data:
            experiment.predict(data)

        mock_filter_data.assert_called_once()
        mock_grouping.group_data.assert_called_once()
        assert mock_predictor.load_model_from_file.call_count == 3
        assert mock_predictor.make_predictions.call_count == 3

    def test__DataExperiment_train__executes_training_workflow(self):
        experiment = DataExperimentTest()

        experiment.train()

    def test__ClinicalOnlyDataExperiment__str__returns_class_name(self):
        assert ClinicalOnlyDataExperiment().__str__() == "ClinicalOnlyDataExperiment"
    
    def test__FullDataExperiment__str__returns_class_name(self):
        assert FullDataExperiment().__str__() == "FullDataExperiment"

    def test__ImagingOnlyDataExperiment__str__returns_class_name(self):
        assert ImagingOnlyDataExperiment().__str__() == "ImagingOnlyDataExperiment"


class DataExperimentTest(DataExperiment):
    def filter_data(self, data):
        return data