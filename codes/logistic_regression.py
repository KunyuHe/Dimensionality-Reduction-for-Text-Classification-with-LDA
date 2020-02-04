import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

class LogisticRegressionVal:
    """
    Logistic Regression classifier with cross validation tuning process.
    """
    LAMBDAS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    PENALTIES = ['l1', 'l2']
    SOLVERS = ['sag', 'saga']

    def __init__(self, X_train, y_train, X_val, y_val, k, random_state=123):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.k = k  # Number of topics from LDA
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


if __name__ == '__main__':
    pass
