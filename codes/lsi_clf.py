from pathlib import Path

import numpy as np
import scipy
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lda_clf import LogisticRegressionVal, evaluate, logger
from utils import loadClean, writeResults, preprocessClfParser

INPUT_DIR = Path(r'../data/clean')


class TruncatedSVD(BaseEstimator, TransformerMixin):
    def __init__(self, k):
        self.k = k

        self.U = None
        self.Sigma = None
        self.VT = None

    def fit(self, X_train):
        if scipy.sparse.issparse(X_train):
            X_train = X_train.toarray()
        if self.VT is None:
            self.U, self.Sigma, self.VT = np.linalg.svd(X_train)
        return self

    def transform(self, X_test):
        if scipy.sparse.issparse(X_test):
            X_test = X_test.toarray()
        proj = self.VT[:self.k, :]
        return X_test @ proj.T


def LSI(train_size, random_state):
    subset = 'subset_%s' % train_size
    input_dir = INPUT_DIR / subset
    K = set((np.linspace(100, train_size - 1, 10) / 100).astype(int) * 100)

    X_train, X_test, y_train, y_test = loadClean(input_dir)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train,
                                                              test_size=0.2,
                                                              random_state=random_state)

    tf_idf = TfidfTransformer()
    X_train_sub = tf_idf.fit_transform(X_train_sub)
    X_val = tf_idf.transform(X_val)
    
    scaler = StandardScaler()

    best_params = []
    best_k, best_auc, best_acc = None, 0, 0
    lsi = TruncatedSVD(k=0)
    lsi.fit(X_train_sub)

    for k in K:
        lsi.k = k
        print(k)
        X_train_ = scaler.fit_transform(lsi.transform(X_train_sub))
        X_val_ = scaler.transform(lsi.transform(X_val))

        clf_val = LogisticRegressionVal(X_train_, y_train_sub, X_val_, y_val,
                                        k, random_state=random_state)
        best_k, best_auc, best_acc, best_params = clf_val.tune(best_k, best_auc,
                                                               best_acc, best_params)

    clf, file_name, header = clf_val.bestClassifier(best_params)
    lsi = TruncatedSVD(k=best_k)  # Create a new one for the whole training set
    preprocess = make_pipeline(tf_idf, lsi, scaler)
    tr_time, tr_metrics, test_time, test_metrics = evaluate(preprocess, clf,
                                                            X_train, y_train,
                                                            X_test, y_test)

    writeResults(file_name, header, 'lsi',
                 train_size, best_k, best_params,
                 tr_time, tr_metrics, test_time, test_metrics)

    logger.info(("\tFor training size = %s, best column dimension k = %s "
                 "best parameter grid: %s (train AUC: {:.3f}, train acc: {:.3f};"
                 " test AUC: {:.3f}, test acc: {:.3f})").
                format(tr_metrics[0], tr_metrics[1],
                       test_metrics[0], test_metrics[1])
                % (train_size, best_k, best_params))


if __name__ == '__main__':
    desc = ("Apply LSI as a preprocessing step, grid search for the best "
            "sub-dimension and hyperparameters.")
    parser = preprocessClfParser(desc)
    args = parser.parse_args()

    if args.all:
        for train_size in np.linspace(1250, 25000, 20):
            LSI(int(train_size), args.random_state)
    else:
        LSI(args.train_size, args.random_state)
