from os import walk

class Experiment:
    def __init__(self):
        self.models = self.get_models()

    def get_models(self):
        if not hasattr(self, 'key'):
            raise TypeError("This method can only be called on a class inherited from Experiment with the `key` property assigned")

        (_, _, model_files) = walk("./models/%s/" %self.key)

class ClinicalOnlyExperiment(Experiment):
