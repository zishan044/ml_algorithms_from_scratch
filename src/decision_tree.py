import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self._feature = feature
        self._threshold = threshold
        self._left = left
        self._right = right
        self._value = value

    def is_leaf_node(self):
        return self._value is not None


class DecisionTree:
    def __init__(self, max_depth=10, min_node_samples=5, n_features=None):
        self.max_depth = max_depth
        self.min_node_samples = min_node_samples
        self.n_features = n_features
        self._root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if self.n_features is None else min(self.n_features, X.shape[1])
        self._root = self._grow_tree(X, y, 0)

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth
            or n_samples <= self.min_node_samples
            or n_labels == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature_ids = np.random.choice(n_features, self.n_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feature_ids)

        if best_feature is None:  # no valid split found
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_ids, right_ids = self._split(X[:, best_feature], best_threshold)
        left_child = self._grow_tree(X[left_ids, :], y[left_ids], depth + 1)
        right_child = self._grow_tree(X[right_ids, :], y[right_ids], depth + 1)

        return Node(best_feature, best_threshold, left_child, right_child)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _best_split(self, X, y, feature_ids):
        best_gain = -1
        split_id, split_threshold = None, None

        for feature_id in feature_ids:
            feature_column = X[:, feature_id]
            thresholds = np.unique(feature_column)

            for threshold in thresholds:
                gain = self._information_gain(y, feature_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_id, split_threshold = feature_id, threshold

        return split_id, split_threshold

    def _information_gain(self, y, feature_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # generate splits
        left_ids, right_ids = self._split(feature_column, threshold)

        if len(left_ids) == 0 or len(right_ids) == 0:
            return 0

        # weighted avg child entropy
        n = len(y)
        e_l = self._entropy(y[left_ids])
        e_r = self._entropy(y[right_ids])
        child_entropy = (len(left_ids) / n) * e_l + (len(right_ids) / n) * e_r

        # information gain
        return parent_entropy - child_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _split(self, column, threshold):
        left_ids = np.argwhere(column <= threshold).flatten()
        right_ids = np.argwhere(column > threshold).flatten()
        return left_ids, right_ids

    def predict(self, X):
        return np.array([self._traverse_tree(x, self._root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node._value
        if x[node._feature] <= node._threshold:
            return self._traverse_tree(x, node._left)
        return self._traverse_tree(x, node._right)


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTree(max_depth=10, min_node_samples=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
