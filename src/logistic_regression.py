import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, alpha=0.001, epochs=10000, tol=0.00001):
        self.alpha = alpha
        self.epochs = epochs
        self.tol = tol
        self.theta = None

    def fit(self, X, y):
        X_b = np.c_[np.ones(shape=(X.shape[0], 1)), X]

        n_samples, n_features = X_b.shape
        self.theta = np.zeros(n_features)

        for i in range(self.epochs):
            grad = (X_b.T @ (sigmoid(X_b @ self.theta) - y))/n_samples
            self.theta -= grad * self.alpha

            if np.linalg.norm(self.theta) < self.tol:
                break
        return self.theta

    def predict(self, X):
        X_b = np.c_[np.ones(shape=(X.shape[0], 1)), X]
        return (sigmoid(X_b @ self.theta) >= 0.5).astype(int)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred)/len(y_test)
    

if __name__ == '__main__':

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(accuracy(y_test, preds))