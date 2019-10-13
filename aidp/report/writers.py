import logging
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support, roc_curve, make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score
from aidp.ml.helpers import specificity, negative_predictive_value, weighted_sensitivity, weighted_ppv, weighted_specificity, weighted_npv, weighted_accuracy

class LogReportWriter():
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._scoring_list={
            'recall':make_scorer(recall_score),
            'precision':make_scorer(precision_score),
            'auc':make_scorer(roc_auc_score),
            'specificity':make_scorer(specificity),
            'npv':make_scorer(negative_predictive_value),
            'accuracy':make_scorer(accuracy_score),
            'weighted_sensitivity':make_scorer(weighted_sensitivity),
            'weighted_ppv':make_scorer(weighted_ppv),
            'weighted_specificity':make_scorer(weighted_specificity),
            'weighted_npv':make_scorer(weighted_npv),
            'weighted_accuracy':make_scorer(weighted_accuracy)
        }

    def write_report(self, model, x_train, y_train, x_test, y_test):
        self._logger.info("")
        self._logger.info("--TRAINING METRICS--")
        for metric in self._scoring_list:
            score = self._scoring_list[metric](model, x_train, y_train)
            self._logger.info("%s\t%s" %(metric, score))
       
        self._logger.info("")
        self._logger.info("--VALIDATION METRICS--")
        for metric in self._scoring_list:            
            score = self._scoring_list[metric](model, x_test, y_test)
            self._logger.info("%s\t%s" %(metric, score))
       
        self._logger.info("")
        