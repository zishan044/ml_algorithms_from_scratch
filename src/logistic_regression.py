import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, alpha=0.001, epochs=10000):
        self.alpha = alpha
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            pred = sigmoid(y_pred)
            
            error = pred - y

            dw = (1/n_samples) * np.dot(X.T, error)
            db = (1/n_samples) * np.sum(error)

            self.weights -= dw * self.alpha
            self.bias -= db * self.alpha

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        pred = sigmoid(y_pred)
        class_pred = (pred > 0.5).astype(int)

        return class_pred


def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred)/len(y_test)
    

if __name__ == '__main__':

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(accuracy(y_test, preds))