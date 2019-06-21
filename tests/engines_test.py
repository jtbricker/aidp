"""Test for the aidp.runners.engines module """
import unittest
from aidp.runners.engines import getEngine, PredictionEngine, TrainingEngine
from aidp.data.modeldata import ModelData

class TestEngineFactory(unittest.TestCase):
    def test_getEngine_predict_returns_PredictionEngine(self):
        key = 'predict'

        engine = getEngine(key, ModelData)

        assert isinstance(engine, PredictionEngine)

    def test_getEngine_predict_returns_TrainngEngine(self):
        key = 'train'

        engine = getEngine(key, ModelData)

        assert isinstance(engine, TrainingEngine)