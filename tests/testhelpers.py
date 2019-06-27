def get_sample_data(tp, tn, fp, fn):
    """ 
    Returns a sample data set with the given number
    of true and false positives and negatives

    Arguments:
        tp {int} -- true postives
        tn {int} -- true negatives
        fp {int} -- false postives
        fn {int} -- false negatives
    
    Returns:
        y_true {array} -- sample "true" data
        y_pred {array} -- sample "predicted" data
    """
    y_pred = [1]*tp + [0]*tn + [1]*fp + [0]*fn
    y_true = [1]*tp + [0]*tn + [0]*fp + [1]*fn

    return y_true, y_pred