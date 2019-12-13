import logging
import os
import time
import scipy
import numpy as np
from pathlib import  Path
import csv
import argparse

PERFORMANCE_DIR = Path(r'../logs/performance')
METRICS_NAMES = ["ROC_AUC", "accuracy", "precision", "recall"]

train_names = ["training " + name for name in METRICS_NAMES]
test_names = ["test " + name for name in METRICS_NAMES]


def createDirs(dir_path):
    """
    Create a new directory if it doesn't exist.
    Inputs:
        - dir_path: (str) path to the directory to create
    Returns:
        (None)
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file = open(dir_path / ".gitkeep", "w+")
    file.close()


def createLogger(log_dir, logger_name):
    createDirs(log_dir)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(log_dir / (time.strftime("%Y%m%d-%H%M%S")+'.log'))
    logger.addHandler(fh)

    return logger


def loadClean(input_dir):
    X_train = scipy.sparse.load_npz(input_dir / "X_train.npz")
    X_test = scipy.sparse.load_npz(input_dir / "X_test.npz")
    y_train = np.load(input_dir / "y_train.npy")
    y_test = np.load(input_dir / "y_test.npy")

    return X_train, X_test, y_train, y_test


def writeResults(file_name,
                 header, preprocess_name,
                 train_cnt, best_k, best_params,
                 train_time, train_metrics,
                 test_time, test_metrics):
    createDirs(PERFORMANCE_DIR)
    file_path = PERFORMANCE_DIR / file_name

    if not os.path.exists(file_path):
        header = ['preprocess', 'train_size', 'k'] + header + \
                 ['train_time'] + train_names + \
                 ['test_time'] + test_names
        with open(file_path, 'w', newline='') as f:
            csv.writer(f).writerow(header)

    with open(file_path, 'a', newline='') as f:
        row = [preprocess_name, train_cnt, best_k] + best_params + \
              [train_time] + train_metrics + \
              [test_time] + test_metrics
        csv.writer(f).writerow(row)


def preprocessClfParser(desc):
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--clf', dest='clf', type=str, default="logistic",
                        help="Choice of classifier.")
    parser.add_argument('--train_size', dest='train_size', type=float,
                        default=1.0, help="Size of used training sample.")
    parser.add_argument('--random_state', dest='random_state', type=int,
                        default=123, help="Seed for algorithms.")
    parser.add_argument('--all', dest='all', type=bool, default=False,
                        help="Whether to run for all train sizes.")

    return parser
