"""
Multi-Layer Perceptron (MLP) implementation based on:
Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors." Nature, 323(6088), 533-536.
"""
import numpy as np
from .base import BaseModel

class MLPModel(BaseModel):
    """
    Multi-Layer Perceptron with one hidden layer, using sigmoid activation and backpropagation.
    Closely follows Rumelhart et al. (1986).
    """
    def __init__(self, config):
        super().__init__(config)
        self.input_size = config["input_size"]  # Number of input units
        self.hidden_size = config["hidden_size"]  # Number of hidden units
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        # Flat weights: [W_ih.flatten(), b_h, W_ho, b_o]
        if "initial_weights" in config and config["initial_weights"]:
            self.set_weights_from_flat(config["initial_weights"])
            print("Loaded initial weights from server (flat)")
        else:
            self.W_ih = np.random.uniform(-0.1, 0.1, (self.input_size, self.hidden_size))
            self.b_h = np.zeros((1, self.hidden_size))
            self.W_ho = np.random.uniform(-0.1, 0.1, (self.hidden_size, 1))
            self.b_o = np.zeros((1, 1))
            print("Using random weight initialization")

    def sigmoid(self, x):
        """Sigmoid activation function (Rumelhart et al., Eq. 1)."""
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        """Forward pass: computes hidden and output activations."""
        H = self.sigmoid(np.dot(X, self.W_ih) + self.b_h)  # Hidden activations
        O = self.sigmoid(np.dot(H, self.W_ho) + self.b_o)  # Output activations
        return O, H

    def train(self, X, y):
        """
        Train using mini-batch SGD and backpropagation (Rumelhart et al. 1986, Algorithm 1).
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
                # Forward pass
                O, H = self.forward(X_batch)
                # Compute error (Eq. 7)
                error = O - y_batch
                # Compute loss (cross-entropy)
                loss = -np.mean(y_batch * np.log(O + 1e-15) + (1 - y_batch) * np.log(1 - O + 1e-15))
                total_loss += loss
                # Backpropagation (Eq. 8-13)
                dO = error * O * (1 - O)  # Output delta
                dW_ho = np.dot(H.T, dO)
                db_o = np.sum(dO, axis=0, keepdims=True)
                dH = np.dot(dO, self.W_ho.T) * H * (1 - H)  # Hidden delta
                dW_ih = np.dot(X_batch.T, dH)
                db_h = np.sum(dH, axis=0, keepdims=True)
                # Update weights (Eq. 14)
                self.W_ih -= self.learning_rate * dW_ih
                self.b_h -= self.learning_rate * db_h
                self.W_ho -= self.learning_rate * dW_ho
                self.b_o -= self.learning_rate * db_o
            if epoch % 2 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")
        self.is_trained = True

    def predict(self, X):
        """Predict class labels (threshold 0.5)."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        O, _ = self.forward(X)
        return (O > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        O, _ = self.forward(X)
        return O.flatten()

    def set_weights_from_flat(self, flat_weights):
        flat = np.array(flat_weights)
        idx = 0
        W_ih_size = self.input_size * self.hidden_size
        self.W_ih = flat[idx:idx+W_ih_size].reshape(self.input_size, self.hidden_size)
        idx += W_ih_size
        b_h_size = self.hidden_size
        self.b_h = flat[idx:idx+b_h_size].reshape(1, self.hidden_size)
        idx += b_h_size
        W_ho_size = self.hidden_size
        self.W_ho = flat[idx:idx+W_ho_size].reshape(self.hidden_size, 1)
        idx += W_ho_size
        self.b_o = flat[idx:idx+1].reshape(1, 1)

    def get_flat_weights(self):
        return np.concatenate([
            self.W_ih.flatten(),
            self.b_h.flatten(),
            self.W_ho.flatten(),
            self.b_o.flatten()
        ]).tolist() 