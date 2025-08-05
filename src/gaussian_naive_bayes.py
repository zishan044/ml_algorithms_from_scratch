import numpy as np

class GaussianNB:
    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)

        self.classes_ = y.values.unique()
        n_classes, n_features = len(self.classes_), X.shape[1]

        self.means_ = np.zeros(shape=(n_classes, n_features))
        self.vars_ = np.zeros(shape=(n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        for idx, k in enumerate(self.classes_):
            X_k = X[y==k]

            self.means_[idx] = X_k.mean(axis=0)
            self.vars_[idx] = X_k.var(axis=0)
            self.priors_[idx] = X_k.shape[0] / X.shape[0]
        
    def _log_probability(self, X):
        num = -0.5 * np.log(((X[:, np.newaxis, :] - self.means_) ** 2) / self.vars_)
        num -= 0.5 * np.log(2 * np.pi * self.vars_)
        return num.sum(axis=2)

    def predict(self, X_test):
        X_test = np.asarray(X_test)

        likelihood = self._log_probability(X_test)
        prior = np.log(self.priors_)

        return