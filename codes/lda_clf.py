import warnings
from pathlib import Path
from time import time

import numpy as np
from joblib import dump, load
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, \
    recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import createLogger, createDirs, loadClean, writeResults, \
    preprocessClfParser

INPUT_DIR = Path(r'../data/clean')
OUTPUT_DIR = Path(r'../logs/models')
LOG_DIR = Path("../logs/pipeline")

logger = createLogger(LOG_DIR, "lda_clf")
logger.info("Logger created, logging to %s" % LOG_DIR.absolute())

K = np.arange(1, 31) * 5
METRICS = [roc_auc_score, accuracy_score, precision_score, recall_score]


class LogisticRegressionVal:
    LAMBDAS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    PENALTIES = ['l1', 'l2']
    SOLVERS = ['sag', 'saga']

    def __init__(self, X_train, y_train, X_val, y_val, k, random_state=123):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.k = k  # # of topics from LDA
        self.lr = LogisticRegression(random_state=random_state)

    def tune(self, best_k, best_auc, best_acc, best_params):
        for lambda_ in self.LAMBDAS:
            for penalty in self.PENALTIES:
                for solver in self.SOLVERS:
                    if penalty == "l1" and solver == "sag":
                        continue
                    self.lr.set_params(**{'C': 1 / lambda_,
                                          'penalty': penalty,
                                          'solver': solver
                                          })
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        self.lr.fit(self.X_train, self.y_train)

                    val_pred = self.lr.predict(self.X_val)
                    val_auc = roc_auc_score(self.y_val, val_pred)
                    val_acc = accuracy_score(self.y_val, val_pred)

                    if (val_auc >= best_auc) and (val_acc > best_acc):
                        best_k, best_auc, best_acc = self.k, val_auc, val_acc
                        best_params = [lambda_, penalty, solver]

        return best_k, best_auc, best_acc, best_params

    def bestClassifier(self, best_params):
        lambda_, penalty, solver = best_params
        self.lr.set_params(**{'C': 1 / lambda_,
                              'penalty': penalty,
                              'solver': solver})

        file_name = "logistic_regression.csv"
        header = ["lambda", "regularizer", "solver"]

        return self.lr, file_name, header


def evaluate(preprocess, clf, X_train, y_train, X_test, y_test):
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


def LDA(train_size, random_state):
    subset = 'subset_%s' % train_size

    input_dir = INPUT_DIR / subset
    model_dir = OUTPUT_DIR / subset
    createDirs(model_dir)

    X_train, X_test, y_train, y_test = loadClean(input_dir)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train,
                                                              test_size=0.2,
                                                              random_state=random_state)
    scaler = StandardScaler()

    best_params = []
    best_k, best_auc, best_acc = None, 0, 0

    for k in K:
        model_name = "lda_%s.joblib" % k
        try:
            lda = load(model_dir / model_name)
            # logger.info("\t\tk = %s, fitted LDA model loaded." % k)
        except:
            lda = LatentDirichletAllocation(n_components=k,
                                            doc_topic_prior=50 / k,
                                            topic_word_prior=0.01,
                                            n_jobs=-1,
                                            random_state=random_state)
            lda.fit(X_train)
            dump(lda, model_dir / model_name)

        X_train_ = scaler.fit_transform(lda.transform(X_train_sub))
        X_val_ = scaler.transform(lda.transform(X_val))

        clf_val = LogisticRegressionVal(X_train_, y_train_sub, X_val_, y_val,
                                        k, random_state=random_state)
        best_k, best_auc, best_acc, best_params = clf_val.tune(best_k, best_auc,
                                                               best_acc, best_params)

    clf, file_name, header = clf_val.bestClassifier(best_params)
    lda.set_params(**{'n_components': best_k,
                      'doc_topic_prior': 50 / best_k})
    preprocess = make_pipeline(lda, scaler)
    tr_time, tr_metrics, test_time, test_metrics = evaluate(preprocess, clf,
                                                            X_train, y_train,
                                                            X_test, y_test)

    writeResults(file_name, header, 'lda',
                 train_size, best_k, best_params,
                 tr_time, tr_metrics, test_time, test_metrics)

    logger.info(("\tFor training size = %s, best number of topics k = %s "
                 "best parameter grid: %s (train AUC: {:.3f}, train acc: {:.3f};"
                 " test AUC: {:.3f}, test acc: {:.3f})").
                format(tr_metrics[0], tr_metrics[1],
                       test_metrics[0], test_metrics[1])
                % (train_size, best_k, best_params))


if __name__ == '__main__':
    desc = ("Apply LDA as a preprocessing step, grid search for best number of "
            "topics and hyperparameters.")
    parser = preprocessClfParser(desc)
    args = parser.parse_args()

    if args.all:
        for train_size in np.linspace(1250, 25000, 20):
            LDA(int(train_size), args.random_state)
    else:
        LDA(args.train_size, args.random_state)
