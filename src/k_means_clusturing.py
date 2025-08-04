import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeansClusturing:
    def __init__(self, k_centoids=3, max_iter=200, threshold=0.0001):
        self.k_centroids = k_centoids
        self.max_iter = max_iter
        self.threshold = threshold

    def fit(self, X):
        centroids = np.random.uniform(
            low=np.min(X, axis=0),
            high=np.max(X, axis=0),
            size=(self.k_centroids, X.shape[1])
        )

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k)
                else np.random.uniform(low=np.min(X, axis=0), high=np.max(X, axis=0))
                for k in range(self.k_centroids)
            ])

            if np.max(np.linalg.norm(centroids-new_centroids, axis=1)) < self.threshold:
                break
            
            centroids = new_centroids
        
        self.centroids = centroids
        self.labels_ = labels

if __name__=="__main__":
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

    kmeans = KMeansClusturing(k_centoids=3)
    kmeans.fit(X)

    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=50, cmap='viridis', alpha=0.6)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
    plt.title("K-Means Clustering Results")
    plt.legend()
    plt.savefig("src/figures/kmeans_output.png")
