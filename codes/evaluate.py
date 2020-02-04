from time import time

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, \
    recall_score

METRICS = [roc_auc_score, accuracy_score, precision_score, recall_score]


def evaluate(preprocess, clf, X_train, y_train, X_test, y_test):
    """
    Fit the optimized preprocessing step and classifier on the training set,
    transform and evaluate on the test set.

    Inputs:
        - preprocess (sklearn.pipeline.Pipeline): preprocessing pipeline
        - clf (sklearn.base.BaseEstimator): classifier
        - X_train (numpy.ndarray): training feature matrix
        - y_train (numpy.ndarray): training target vector
        - X_test (numpy.ndarray): test feature matrix
        - y_test (numpy.ndarray): test target vector

    Output:
        (float, [float], float, [float]) number of seconds took for training,
            evaluations on the training set, number of seconds took for testing,
             evaluations on the test set
    """
    train_start = time()
    preprocess.fit(X_train)
    X_train_ = preprocess.transform(X_train)
    clf.fit(X_train_, y_train)
    train_time = "%0.3fs" % (time() - train_start)
    train_pred = clf.predict(X_train_)
    train_metrics = [metric(y_train, train_pred) for metric in METRICS]

    test_start = time()
    X_test_ = preprocess.transform(X_test)
    test_pred = clf.predict(X_test_)
    test_time = "%0.3fs" % (time() - test_start)
    test_metrics = [metric(y_test, test_pred) for metric in METRICS]

    return train_time, train_metrics, test_time, test_metrics


if __name__ == '__main__':
    pass
