import numpy as np


class MaskedImageModel:
    """
    Neural network for predicting a hidden 8x8 image region.

    Architecture:
        784 inputs → 128 hidden neurons → 64 outputs
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=64, seed=42):
        rng = np.random.default_rng(seed)

        # Input → hidden layer
        self.W1 = rng.normal(
            0,
            np.sqrt(2 / input_size),
            (input_size, hidden_size)
        )
        self.b1 = np.zeros(hidden_size)

        # Hidden → output layer
        self.W2 = rng.normal(
            0,
            np.sqrt(2 / hidden_size),
            (hidden_size, output_size)
        )
        self.b2 = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        """
        Forward pass.

        X shape:
            (number_of_images, 784)

        Returns:
            predictions with shape (number_of_images, 64)
        """

        # Input → hidden
        self.z1 = X @ self.W1 + self.b1

        # Activation
        self.a1 = self.relu(self.z1)

        # Hidden → output
        self.z2 = self.a1 @ self.W2 + self.b2

        # Predicted hidden pixels
        self.output = self.z2

        return self.output

    def mse_loss(self, predictions, targets):
        """
        Mean Squared Error between predicted
        and actual hidden pixels.
        """

        return np.mean((predictions - targets) ** 2)

    def backward(self, X, targets, learning_rate=0.001):
        """
        Backpropagation and parameter update.
        """

        batch_size = X.shape[0]

        # Error at output
        dZ2 = (2 / (batch_size * targets.shape[1])) * (
            self.output - targets
        )

        # Gradients for W2 and b2
        dW2 = self.a1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)

        # Propagate error backward
        dA1 = dZ2 @ self.W2.T

        # Through ReLU
        dZ1 = dA1 * self.relu_derivative(self.z1)

        # Gradients for W1 and b1
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        # Update parameters
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1