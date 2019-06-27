"""
The :mod:`mlexp.nbutils` module implements severals methods that are used for machine
learning experiments in jupyter notebooks.
"""
import itertools
import os

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, roc_curve, classification_report
import numpy as np
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
   print('no display found. Using non-interactive Agg backend')
   mpl.use('Agg')
import matplotlib.pyplot as plt

def specificity(y_true, y_pred):
    """ Calculates the specificity (Selectivity, True Negative Rate)
    
    Arguments:
        y_true {array-like} -- true classes
        y_pred {array-like} -- predicted classes
    
    Returns:
        float -- specificity
    """

    cm = confusion_matrix(y_true, y_pred)
    return cm[0,0] / cm[0,:].sum()

def negative_predictive_value(y_true, y_pred):
    """ Calculates the negative predictive value
    
    Arguments:
        y_true {array-like} -- true classes
        y_pred {array-like} -- predicted classes
    
    Returns:
        float -- negative_predictive_value
    """

    cm = confusion_matrix(y_true, y_pred)
    return cm[0,0] / cm[:,0].sum()

def get_weighted_confusion_matrix(y_true, y_pred):
    """ Calculates the confusion matrix weighted by
    class size
    
    Arguments:
        y_true {array-like} -- true classes
        y_pred {array-like} -- predicted classes
    
    Returns:
        tp_weighted -- weighted true positives
        fp_weighted -- weighted false positives
        fn_weighted -- weighted false negatives
        tn_weighted -- weighted true negatives
    """

    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0,0]
    tp = cm[1,1]
    fp = cm[0,1]
    fn = cm[1,0]
    
    tp_weighted = tp / (tp + fn)
    fp_weighted = fp / (tn + fp)
    fn_weighted = fn / (tp + fn)
    tn_weighted = tn / (tn + fp)
    
    return tp_weighted, fp_weighted, fn_weighted, tn_weighted

def weighted_accuracy(y_true, y_pred):
    """ Calculates the weighted accuracy of the predictions
    
    Arguments:
        y_true {array-like} -- true classes
        y_pred {array-like} -- predicted classes
    
    Returns:
        float -- weighted_accuracy
    """

    tpw, fpw, fnw, tnw = get_weighted_confusion_matrix(y_true, y_pred)
    
    return (tpw + tnw) / (tpw + fpw + fnw + tnw)
    
def weighted_sensitivity(y_true, y_pred):
    """ Calculates the weighted sensitivity (aka True Postiive Rate, Recall) of the predictions
    
    Arguments:
        y_true {array-like} -- true classes
        y_pred {array-like} -- predicted classes
    
    Returns:
        float -- weighted_sensitivity
    """

    tpw, _, fnw, _ = get_weighted_confusion_matrix(y_true, y_pred)

    return tpw / (tpw + fnw)
    
def weighted_specificity(y_true, y_pred):
    """ Calculates the weighted specificity (aka Selectivity, True Negative Rate) of the predictions
    
    Arguments:
        y_true {array-like} -- true classes
        y_pred {array-like} -- predicted classes
    
    Returns:
        float -- weighted_specificity
    """

    _, fpw, _, tnw = get_weighted_confusion_matrix(y_true, y_pred)

    return tnw / (tnw + fpw)
    
def weighted_ppv(y_true, y_pred):
    """ Calculates the weighted positive predictive value (aka Precision) of the predictions
    
    Arguments:
        y_true {array-like} -- true classes
        y_pred {array-like} -- predicted classes
    
    Returns:
        float -- weighted_ppv
    """

    tpw, fpw, _, _ = get_weighted_confusion_matrix(y_true, y_pred)
    
    return tpw / (tpw + fpw)

def weighted_npv(y_true, y_pred):
    """ Calculates the weighted negative predictive value of the predictions
    
    Arguments:
        y_true {array-like} -- true classes
        y_pred {array-like} -- predicted classes
    
    Returns:
        float -- weighted_npv
    """

    _, _, fnw, tnw = get_weighted_confusion_matrix(y_true, y_pred)
    
    return tnw / (tnw + fnw)

def print_score_summaries(scores_dict):
    """ Prints out the mean and stddev of scores, dropping any NaN values in the calculation
    
    Arguments:
        scores_dict {dict(name(str) -> scores(float))} -- A dictionary that maps the name of a score value (e.g. "specificity")
        to an array of those scores
    """

    for score_name in scores_dict:
        scores = scores_dict[score_name]
        print("%s\t%s\t%s" %(score_name, np.mean(scores[~np.isnan(scores)]), np.std(scores[~np.isnan(scores)])))

def get_metrics(model, X, y, scoring_list={'accuracy':make_scorer(accuracy_score)}):
    """Get a dictionary of calculated metrics given a model and known data
    
    Arguments:
        model {sklearn Estimator} -- A fitted sklearn estimator 
        X {array-like(float)} -- data's features
        y {int} -- data's class
        scoring_list {dict(scorer_name(str) -> sklearn scorer)} -- dictionary of all scores you would like
        to calculate on the data/model combination

    Returns:
        dict -- scorer name (str) -> score (float) 
    """

    metrics = {}
    for metric in scoring_list:
        score = scoring_list[metric](model, X, y)
        metrics[metric] = score
        print("%s\t%s" %(metric, score))

    return metrics

def plot_roc(model, X_test, Y_test, verbose=False, show_plot=True):
    """Diplays the roc curve given the model and test data
    
    Arguments:
        model {sklearn estimator} -- fitted estimator
        X_test {array-like} -- data features
        Y_test {array-like} -- data classes
    
    Keyword Arguments:
        verbose {bool} -- [If true, diplays optional classification information and raw data] (default: {False})
    """

    y_true, y_pred = Y_test, model.predict(X_test)
    if verbose:
        print("CLASSIFICATION REPORT")
        print(classification_report(y_true, y_pred))

    y_pred_prob = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)

    if verbose:
        print("TESTING PROBABILITIES:")
        for a,b in zip(Y_test,y_pred_prob):
            print(a,b)
    
    if show_plot:
        plt.plot([0,1],[0,1], 'k--')
        plt.plot(fpr, tpr, label='Linear SVC')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Linear SVC ROC Curve')
        plt.show(block=False)
    
    if verbose:
        print("ROC RAW DATA:")
        for a,b in zip(fpr, tpr):
            print(a,b)

def plot_coefficients(classifier, feature_names, top_features=20, show_plot=True):
    """Creates a barplot of the top most important features
    
    Arguments:
        classifier {sklearn estimator} -- fitted estimator
        feature_names {array-like(str)} -- list of names of the features to display in plot
    
    Keyword Arguments:
        top_features {int} -- The number of features to display on chart (default: {20})
    """

    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    if show_plot:
        # create plot
        plt.figure(figsize=(15, 5))
        colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
        feature_names = np.array(feature_names)
        plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
        plt.show()

def print_feature_importance(feature_names, coefs):
    """Helper method to pair and print feature name/important
    
    Arguments:
        feature_names {array-like(str)} -- feature names
        coefs {array-like(float)} -- feature coeficients
    """

    assert len(feature_names) == len(coefs), "Arrays have difference lengths. Something went wrong"
    for feature, coef in zip(feature_names, coefs):
        print("%s\t%s" %(feature, coef))

def plot_confusion_matrix(cm, classes=[0,1], normalize=False, title='Confusion matrix', print_matrix=False, show_plot=True):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Arguments:
        cm {numpy array} -- confusion matrix
    
    Keyword Arguments:
        classes {list} -- class labels (default: {[0,1]})
        normalize {bool} -- should the confusion matrix be normalized (default: {False})
        title {str} -- plot title (default: {'Confusion matrix'})
        print_matrix {bool} -- should the raw confusion matrix be printed (default: {False})
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    if print_matrix:
        print(cm)

    if show_plot:
        plt.imshow(cm, interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

def grid_search_optimization(pipeline, parameters_to_tune, X, y, Xh, yh, cv=5, scoring='accuracy', verbose=False, n_jobs=-1):
    """ Performs a grid-search optimization with cross validation with the provided hyperparameters
    and outputs a report
    
    Arguments:
        pipeline {sklearn pipeline} -- unfitted pipeline of transformations and estimator
        parameters_to_tune {dict(named parameter(str) -> array-like (any))} -- a dictionary that maps a 
        X {array-like} -- data - features to fit
        y {array-like} -- data - classes to fit
        Xh {array-like} -- holdout data - features to test
        yh {array-like} -- holdout data - classes to test
    
    Keyword Arguments:
        cv {int} -- number of cross-validation folds (default: {5})
        n_jobs {int} -- number of cores to use in optimization (-1 is all available) (default: {-1})
        scoring {str} -- name of scorer to use for optimization (default: {'accuracy'})
        verbose {bool} -- show additional information? (default: {False})
    
    Returns:
        [sklearn estimator] -- fitted pipeline
    """

    print("# Tuning hyper-parameters for %s" %scoring)
    print()

    clf = GridSearchCV(pipeline, parameters_to_tune, cv=cv, n_jobs = n_jobs, scoring=scoring, verbose=verbose)
    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()

    if verbose:
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report (holdout):")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = yh, clf.predict(Xh)
        print(classification_report(y_true, y_pred))
        print()
        plot_confusion_matrix(confusion_matrix(y_true, y_pred)) 
        print()
        get_metrics(clf.best_estimator_, Xh, yh)
    
        print("TRAINNG PROBABILITIES")
        for a,b in zip(y, clf.predict_proba(X)[:,1]):
            print(a,b)
    
    return clf