"""
Logistic Regression implementation based on:
Cox, D. R. (1958). "The regression analysis of binary sequences." Journal of the Royal Statistical Society: Series B (Methodological), 20(2), 215-242.
"""
import numpy as np
from .base import BaseModel

class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression for binary classification using sigmoid activation.
    Closely follows Cox (1958).
    """
    def __init__(self, config):
        super().__init__(config)
        self.input_size = config["input_size"]
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        # Flat weights: [beta..., beta_0]
        if "initial_weights" in config and config["initial_weights"]:
            self.set_weights_from_flat(config["initial_weights"])
            print("Loaded initial weights from server (flat)")
        else:
            self.beta = np.random.uniform(-0.1, 0.1, (self.input_size, 1))
            self.beta_0 = 0.0
            print("Using random weight initialization")

    def sigmoid(self, x):
        """Sigmoid function as in Cox (1958)."""
        return 1 / (1 + np.exp(-x))

    def train(self, X, y):
        """
        Train using mini-batch SGD and the log-likelihood gradient (Cox, 1958).
        """
        n_samples = X.shape[0]
        for epoch in range(self.epochs):
            perm = np.random.permutation(n_samples)
            X_shuffled = X[perm]
            y_shuffled = y[perm].reshape(-1, 1)
            total_loss = 0
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                # Linear predictor
                z = np.dot(X_batch, self.beta) + self.beta_0
                # Sigmoid output
                p = self.sigmoid(z)
                # Log-likelihood loss
                loss = -np.mean(y_batch * np.log(p + 1e-15) + (1 - y_batch) * np.log(1 - p + 1e-15))
                total_loss += loss
                # Gradient (Cox, Eq. 2)
                error = p - y_batch
                grad_beta = np.dot(X_batch.T, error) / X_batch.shape[0]
                grad_beta_0 = np.mean(error)
                # Update coefficients
                self.beta -= self.learning_rate * grad_beta
                self.beta_0 -= self.learning_rate * grad_beta_0
            if epoch % 2 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")
        self.is_trained = True

    def predict(self, X):
        """Predict class labels (threshold 0.5)."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        z = np.dot(X, self.beta) + self.beta_0
        p = self.sigmoid(z)
        return (p > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        z = np.dot(X, self.beta) + self.beta_0
        p = self.sigmoid(z)
        return p.flatten()

    def set_weights_from_flat(self, flat_weights):
        flat = np.array(flat_weights)
        self.beta = flat[:-1].reshape(-1, 1)
        self.beta_0 = float(flat[-1])

    def get_flat_weights(self):
        return np.concatenate([self.beta.flatten(), [self.beta_0]]).tolist() 