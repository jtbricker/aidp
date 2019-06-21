"""Test for the aidp.runners.engines module """
import unittest
from unittest.mock import Mock
from aidp.runners.engines import getEngine, PredictionEngine, TrainingEngine
from aidp.data.modeldata import ModelData

class TestEngineFactory(unittest.TestCase):
    def test__getEngine_predict__returns_PredictionEngine(self):
        key = 'predict'

        engine = getEngine(key, None)

        assert isinstance(engine, PredictionEngine)

    def test__getEngine_train__returns_TrainingEngine(self):
        key = 'train'

        engine = getEngine(key, None)

        assert isinstance(engine, TrainingEngine)

    def test__getEngine_unsupported_key__throws_error(self):
        key = 'garbage'

        with self.assertRaises(NotImplementedError):
            getEngine(key, ModelData)

    def test__PredictionEngine_start__calls_engine_predict(self):
        mock_model_data = Mock()
        mock_experiment = Mock()
        engine = PredictionEngine(mock_model_data)
        engine.experiments = [mock_experiment]

        engine.start()

        mock_experiment.predict.assert_called()

    def test__TrainingEngine_start__calls_engine_train(self):
        mock_model_data = Mock()
        mock_experiment = Mock()
        engine = TrainingEngine(mock_model_data)
        engine.experiments = [mock_experiment]

        engine.start()

        mock_experiment.train.assert_called()
