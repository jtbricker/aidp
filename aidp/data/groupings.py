"""This module defines the different data using different groupings.

    Grouping are defined by which group IDs are considered the negative
    class, which ones are considered the postive class, and which group IDs
    are not considered at all.
 """

from abc import ABC, abstractmethod

class DataGrouping(ABC):
    postive_groups = []
    negative_groups = []

    def group_data(self, data):        
        grouping = {
            **{ k: 0 for k in self.negative_groups},
            **{ k: 1 for k in self.postive_groups}
        }

        self.grouped_data = reassign_classes(data, grouping, "GroupID")
        return self


class ParkinsonsVsControlGrouping(DataGrouping):
    def __init__(self):
        self.key = "park_v_control"

        self.positive_label = "PD/MSA/PSP"
        self.postive_groups = [1,2,3]
        
        self.negative_label = "Control"
        self.negative_groups = [0]

class MsaPspVsPdGrouping(DataGrouping):
    def __init__(self):
        self.key = "msa_psp_v_pd"

        self.positive_label = "MSA/PSP"
        self.postive_groups = [2,3]
        
        self.negative_label = "PD"
        self.negative_groups = [1]

class MsaVsPdPspGrouping(DataGrouping):
    def __init__(self):
        self.key = "msa_v_pd_psp"

        self.positive_label = "MSA"
        self.postive_groups = [2]

        self.negative_groups = "PD/PSP"
        self.negative_groups = [1,3]

class PspVsPdMsaGrouping(DataGrouping):
    def __init__(self):
        self.key = "psp_v_pd_msa"

        self.positive_label = "PSP"
        self.postive_groups = [3]

        self.negative_groups = "PD/MSA"
        self.negative_groups = [1,2]

class PspVsMsaGrouping(DataGrouping):
    def __init__(self):
        self.key = "psp_v_msa"
        
        self.positive_label = "PSP"
        self.postive_groups = [3]

        self.negative_groups = "MSA"
        self.negative_groups = [2]

def reassign_classes(data, grouping, group_col):
        """
        Returns a subset of the data with new class labels
        ----------
        data : DataFrame
        grouping : dict, keys = classes to keep, values = new labels of classes
        group_col: string - column name to group on
        
        Returns
        -------
        data_subset : DataFrame
            subset of data, where only rows with class labels that are keys in grouping 
            are kept and whose new class labels are the corresponding values in grouping
        """
        classes_to_keep = grouping.keys()
        data_to_keep = data.loc[data[group_col].isin(classes_to_keep)]
        classes_to_change = {k:grouping[k] for k in classes_to_keep if k!= grouping[k]}
        return data_to_keep.replace(classes_to_change)