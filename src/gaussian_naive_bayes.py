import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class GaussianNB:
    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)

        self.classes_ = np.unique(y)
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
        num = -0.5 * ((X[:, np.newaxis, :] - self.means_) ** 2) / self.vars_
        num -= 0.5 * np.log(2 * np.pi * self.vars_)
        return num.sum(axis=2)

    def predict(self, X_test):
        X_test = np.asarray(X_test)

        log_likelihood = self._log_probability(X_test)
        log_prior = np.log(self.priors_)

        return self.classes_[np.argmax(log_likelihood + log_prior, axis=1)]
    
if __name__=="__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(accuracy_score(y_pred=y_pred, y_true=y_test))