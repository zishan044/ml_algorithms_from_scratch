import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter

class KNN:
    def __init__(self, k_neigbors=7):
        self.k_neighbors = k_neigbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def label(self, x_instance):
        distances = [np.linalg.norm(x_instance-self.X_train[i]) for i in range(len(self.X_train))]
        k_indices = np.argsort(distances)[:self.k_neighbors]
        return Counter(self.y_train[k_indices]).most_common(1)[0][0]
    
    def predict(self, X_test):
        return [self.label(X_test[i]) for i in range(len(X_test))]
    

def accuracy(y_preds, y_test):
    return (y_preds == y_test).sum()/len(y_test)

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KNN()
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    print(f"Accuracy: {accuracy(y_preds, y_test):.4f}")