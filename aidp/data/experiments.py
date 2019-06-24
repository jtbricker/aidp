"""This module defines the different data experiments.

    Experiments are defined by what data is used when creating models.
    Some subset of the input data is used for each experiment.
 """
import logging
from abc import ABC, abstractmethod
from aidp.data.groupings import ParkinsonsVsControlGrouping, MsaPspVsPdGrouping, MsaVsPdPspGrouping, PspVsPdMsaGrouping, PspVsMsaGrouping
from aidp.ml.predictors import Predictor

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

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    @abstractmethod
    def filter_data(self, data):
        pass #pragma: no cover

    def predict(self, data):
        self._logger.info("Starting model prediction")
        self.filtered_data = self.filter_data(data)
        for grouping in self.groupings:
            self.predictor.load_model_from_file(self.key, grouping.key)
            grouping.predictions = self.predictor.make_predictions(self.filtered_data)
        self._logger.info("Starting model prediction")

    def train(self):
        self._logger.info("Starting model training")
        #TODO: Implement Training mechanism
        self._logger.debug("Finished model training")       

    def __str__(self):
        return type(self).__name__

class ClinicalOnlyDataExperiment(DataExperiment):
    key = "updrs-plus" #

    def filter_data(self, data):
        standard_data = get_standardized_data(data)
        return standard_data[['GroupID', 'Age', 'Sex', 'UPDRS']]

class ImagingOnlyDataExperiment(DataExperiment):
    key = "no-updrs" #TODO: Call this dmri

    def filter_data(self, data):
        standard_data = get_standardized_data(data)
        return standard_data.drop(['UPDRS'], axis=1)

class FullDataExperiment(DataExperiment):
    key = "both"

    def filter_data(self, data):
        return get_standardized_data(data)


def get_standardized_data(data):
    #TODO: Get this list of strings from a config file
    return data[[
        'GroupID',
        'Age',
        'Sex',
        'UPDRS',
        'aSN_FA',
        'Caudate_FA',
        'CC2_FA',
        'GP_FA',
        'LobuleVI_FA',
        'LobuleV_FA',
        'MCP_FA',
        'pSN_FA',
        'Putamen_FA',
        'SCP_FA',
        'STN_FA',
        'Thalamus_FA',
        'Vermis_FA',
        'RN_FA',
        'PPN_FA',
        'Dentate_FA',
        'CC1_FA',
        'aSN_FW',
        'Caudate_FW',
        'CC2_FW',
        'GP_FW',
        'LobuleVI_FW',
        'LobuleV_FW',
        'MCP_FW',
        'pSN_FW',
        'Putamen_FW',
        'SCP_FW',
        'STN_FW',
        'Thalamus_FW',
        'Vermis_FW',
        'RN_FW',
        'PPN_FW',
        'Dentate_FW',
        'CC1_FW',
        'Angular_Gyrus_Final_FA',
        'Anterior_Orbital_Gyrus_Final_FA',
        'Calcarine_Sulcus_Final_FA',
        'Cuneus_Final_FA',
        'Gyrus_Rectus_Final_FA',
        'Inferior_Frontal_Gyrus_Pars_Opercularis_Final_FA',
        'Inferior_Frontal_Gyrus_Pars_Orbitalis_Final_FA',
        'Inferior_Frontal_Gyrus_Pars_Triangularis_Final_FA',
        'Inferior_Occipital_Final_FA',
        'Inferior_Parietal_Lobule_Final_FA',
        'Inferior_Temporal_Gyrus_Final_FA',
        'Lateral_Orbital_Gyrus_Final_FA',
        'Lingual_Gyrus_Final_FA',
        'M1_Final_FA',
        'Medial_Frontal_Gyrus_Final_FA',
        'Medial_Orbital_Gyrus_Final_FA',
        'Medial_Orbitofrontal_Gyrus_Final_FA',
        'Middle_Frontal_Gyrus_Final_FA',
        'Middle_Occipital_Final_FA',
        'Middle_Temporal_Gyrus_Final_FA',
        'Olfactory_Cortex_Final_FA',
        'Paracentral_Final_FA',
        'PMd_Final_FA',
        'PMv_Final_FA',
        'preSMA_Final_FA',
        'S1_Final_FA',
        'SMA_Final_FA',
        'Superior_Frontal_Gyrus_Final_FA',
        'Superior_Occipital_Final_FA',
        'Superior_Parietal_Lobule_Final_FA',
        'Superior_Temporal_Gyrus_Final_FA',
        'Supramarginal_Gyrus_Final_FA',
        'M1_SMATT_FA',
        'PMd_SMATT_FA',
        'PMv_SMATT_FA',
        'SMA_SMATT_FA',
        'preSMA_SMATT_FA',
        'S1_SMATT_FA',
        'Cerebellar_MCP_FA',
        'Cerebellar_SCP_FA',
        'Nigrostriatal_FA',
        'STN_to_GP_FA',
        'Corticostriatal_FA',
        'Angular_Gyrus_Final_FW',
        'Anterior_Orbital_Gyrus_Final_FW',
        'Calcarine_Sulcus_Final_FW',
        'Cuneus_Final_FW',
        'Gyrus_Rectus_Final_FW',
        'Inferior_Frontal_Gyrus_Pars_Opercularis_Final_FW',
        'Inferior_Frontal_Gyrus_Pars_Orbitalis_Final_FW',
        'Inferior_Frontal_Gyrus_Pars_Triangularis_Final_FW',
        'Inferior_Occipital_Final_FW',
        'Inferior_Parietal_Lobule_Final_FW',
        'Inferior_Temporal_Gyrus_Final_FW',
        'Lateral_Orbital_Gyrus_Final_FW',
        'Lingual_Gyrus_Final_FW',
        'M1_Final_FW',
        'Medial_Frontal_Gyrus_Final_FW',
        'Medial_Orbital_Gyrus_Final_FW',
        'Medial_Orbitofrontal_Gyrus_Final_FW',
        'Middle_Frontal_Gyrus_Final_FW',
        'Middle_Occipital_Final_FW',
        'Middle_Temporal_Gyrus_Final_FW',
        'OlFWctory_Cortex_Final_FW',
        'Paracentral_Final_FW',
        'PMd_Final_FW',
        'PMv_Final_FW',
        'preSMA_Final_FW',
        'S1_Final_FW',
        'SMA_Final_FW',
        'Superior_Frontal_Gyrus_Final_FW',
        'Superior_Occipital_Final_FW',
        'Superior_Parietal_Lobule_Final_FW',
        'Superior_Temporal_Gyrus_Final_FW',
        'Supramarginal_Gyrus_Final_FW',
        'M1_SMATT_FW',
        'PMd_SMATT_FW',
        'PMv_SMATT_FW',
        'SMA_SMATT_FW',
        'preSMA_SMATT_FW',
        'S1_SMATT_FW',
        'Cerebellar_MCP_FW',
        'Cerebellar_SCP_FW',
        'Nigrostriatal_FW',
        'STN_to_GP_FW',
        'Corticostriatal_FW'
    ]]
