import numpy as np

def gradient_descent(X, y, m, b, alpha=0.01):
    n = len(X)
    y_pred = m * X + b
    error = y_pred - y
    m -= (2/n) * np.sum(error * X) * alpha
    b -= (2/n) * np.sum(error) * alpha
    return m, b

if __name__ == "__main__":
    X = np.random.randint(1, 100, size=(1000))
    X = (X - X.mean()) / X.std()  # Feature scaling

    y = 4 * X + 3 + np.random.normal(0, 5, size=X.shape)

    m, b = 0, 0

    for i in range(1000):
        m, b = gradient_descent(X, y, m, b, alpha=0.01)

    print(f"m: {m:.4f}\nb: {b:.4f}")
