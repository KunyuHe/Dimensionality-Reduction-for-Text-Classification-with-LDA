from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from lda_clf import LogisticRegressionVal, evaluate, logger
from utils import loadClean, writeResults, preprocessClfParser

INPUT_DIR = Path(r'../data/clean')

def CLF(train_size, random_state):
    subset = 'subset_%s' % train_size
    input_dir = INPUT_DIR / subset

    X_train, X_test, y_train, y_test = loadClean(input_dir)
    X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train,
                                                              test_size=0.2,
                                                              random_state=random_state)

    tf_idf = TfidfTransformer()
    X_train_ = tf_idf.fit_transform(X_train_)
    X_val = tf_idf.transform(X_val)

    clf_val = LogisticRegressionVal(X_train_, y_train_, X_val, y_val,
                                    "NA", random_state=random_state)
    best_k, best_auc, best_acc, best_params = clf_val.tune("NA", 0, 0, [])
    clf, file_name, header = clf_val.bestClassifier(best_params)
    preprocess = make_pipeline(tf_idf)
    tr_time, tr_metrics, test_time, test_metrics = evaluate(preprocess, clf,
                                                            X_train, y_train,
                                                            X_test, y_test)

    writeResults(file_name, header, 'tf-idf',
                 train_size, best_k, best_params,
                 tr_time, tr_metrics, test_time, test_metrics)

    logger.info(("\tFor training size = %s best parameter grid: %s (train AUC: "
                 "{:.3f}, train acc: {:.3f}; test AUC: {:.3f}, test acc: {:.3f})").
                format(tr_metrics[0], tr_metrics[1],
                       test_metrics[0], test_metrics[1])
                % (train_size, best_params))


if __name__ == '__main__':
    desc = ("Apply tf-idf alone as a preprocessing step, grid search for the"
            " best set of hyperparameters.")
    parser = preprocessClfParser(desc)
    args = parser.parse_args()

    if args.all:
        for train_size in np.linspace(1250, 25000, 20):
            CLF(int(train_size), args.random_state)
    else:
        CLF(args.train_size, args.random_state)
