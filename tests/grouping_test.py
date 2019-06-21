"""Test for the aidp.data.grouping module """
import unittest
from aidp.data.groupings import reassign_classes, ParkinsonsVsControlGrouping, MsaPspVsPdGrouping, MsaVsPdPspGrouping, PspVsPdMsaGrouping, PspVsMsaGrouping
import pandas as pd

class TestGroupings(unittest.TestCase):
    def test_reassign_classes_keep_all_data_but_reassign(self):
        """Return all data but reassign the labels"""
        data = pd.DataFrame([{'class':0,'data':100},{'class':1,'data':200},{'class':0,'data':500}])
        new_data = reassign_classes(data, {0:1,1:0}, 'class')

        assert len(data.index) == len(new_data.index)
        assert new_data['class'].value_counts()[0] == data['class'].value_counts()[1]
        assert new_data['class'].value_counts()[1] == data['class'].value_counts()[0]

    def test_reassign_classes_keep_only_one_class(self):
        """Return all data but reassign the labels"""
        data = pd.DataFrame([{'class':0,'data':100},{'class':1,'data':200},{'class':0,'data':500}])
        new_data = reassign_classes(data, {0:0}, 'class')

        assert len(new_data.index) == len(data[data['class'] == 0])
        assert new_data['class'].value_counts()[0] == data['class'].value_counts()[0]

    def test_reassign_classes_three_groups_into_two(self):
        """Return all data but reassign the labels"""
        data = pd.DataFrame([{'class':0,'data':100},{'class':1,'data':200},{'class':0,'data':500},{'class':2,'data':400}])
        new_data = reassign_classes(data, {0:0, 1:1, 2:1}, 'class')

        assert new_data['class'].value_counts()[0] == data['class'].value_counts()[0]
        assert new_data['class'].value_counts()[1] == data['class'].value_counts()[1] + data['class'].value_counts()[2]

    def test__ParkinsonsVsControlGrouping__group_data(self):
        """Test groupings for Parkinsons (positve) vs Control (negative)"""
        data = get_test_data()

        grouping = ParkinsonsVsControlGrouping().group_data(data)

        assert grouping.grouped_data['GroupID'].value_counts()[0] == 1
        assert grouping.grouped_data['GroupID'].value_counts()[1] == 9

    def test__MsaPspVsPdGrouping__group_data(self):
        """Test groupings for MSA/PSP (positve) vs PD (negative)"""
        data = get_test_data()

        grouping = MsaPspVsPdGrouping().group_data(data)

        assert grouping.grouped_data['GroupID'].value_counts()[0] == 2
        assert grouping.grouped_data['GroupID'].value_counts()[1] == 7

    def test__MsaVsPdPspGrouping__group_data(self):
        """Test groupings for MSA (positve) vs PD/PSP (negative)"""
        data = get_test_data()

        grouping = MsaVsPdPspGrouping().group_data(data)

        assert grouping.grouped_data['GroupID'].value_counts()[0] == 6
        assert grouping.grouped_data['GroupID'].value_counts()[1] == 3

    def test__PspVsPdMsaGrouping__group_data(self):
        """Test groupings for PSP (positve) vs PD/MSA (negative)"""
        data = get_test_data()

        grouping = PspVsPdMsaGrouping().group_data(data)

        assert grouping.grouped_data['GroupID'].value_counts()[0] == 5
        assert grouping.grouped_data['GroupID'].value_counts()[1] == 4

    def test__PspVsMsaGrouping__group_data(self):
        """Test groupings for PSP (positve) vs MSA (negative)"""
        data = get_test_data()

        grouping = PspVsMsaGrouping().group_data(data)

        assert grouping.grouped_data['GroupID'].value_counts()[0] == 3
        assert grouping.grouped_data['GroupID'].value_counts()[1] == 4


def get_test_data():
    return pd.DataFrame([
        {'GroupID':0,'data':100},
        {'GroupID':1,'data':200},{'GroupID':1,'data':200},
        {'GroupID':2,'data':500},{'GroupID':2,'data':500},{'GroupID':2,'data':500},
        {'GroupID':3,'data':400},{'GroupID':3,'data':400},{'GroupID':3,'data':400},{'GroupID':3,'data':400}
    ])



