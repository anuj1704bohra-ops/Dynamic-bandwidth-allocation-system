"""Forecasting models for traffic demand prediction."""

from __future__ import annotations

import numpy as np
try:
    from src.constants import CLASS_NAMES, HISTORY_LENGTH
except ImportError:
    from constants import CLASS_NAMES, HISTORY_LENGTH


class LastObservedPredictor:
    """Persistence baseline: forecasts next demand as the most recent observed demand."""

    def predict(self, history: np.ndarray) -> np.ndarray:
        """history shape: (N, HISTORY_LENGTH * NUM_CLASSES) or (HISTORY_LENGTH, NUM_CLASSES)"""
        k = len(CLASS_NAMES)
        arr = np.asarray(history, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # Reshape to (N, HISTORY_LENGTH, NUM_CLASSES) and take the last slot
        reshaped = arr.reshape(-1, HISTORY_LENGTH, k)
        return reshaped[:, -1, :].copy()


class MovingAveragePredictor:
    """Moving-average baseline: forecasts next demand as the mean of recent history slots."""

    def __init__(self, window: int = HISTORY_LENGTH):
        self.window = window

    def predict(self, history: np.ndarray) -> np.ndarray:
        """history shape: (N, HISTORY_LENGTH * NUM_CLASSES) or (HISTORY_LENGTH, NUM_CLASSES)"""
        k = len(CLASS_NAMES)
        arr = np.asarray(history, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        reshaped = arr.reshape(-1, HISTORY_LENGTH, k)
        w = min(self.window, HISTORY_LENGTH)
        return reshaped[:, -w:, :].mean(axis=1)


class OraclePredictor:
    """Ideal reference: knows the true actual demand at time slot t."""

    def predict(self, actual_demand: np.ndarray) -> np.ndarray:
        arr = np.asarray(actual_demand, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.copy()


class DemandMLP:
    """Feed-forward neural network with two ReLU hidden layers trained via mini-batch Adam."""

    def __init__(self, input_size: int, output_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He initialization for ReLU layers
        self.w1 = rng.normal(0, np.sqrt(2.0 / input_size), (input_size, 48))
        self.b1 = np.zeros(48, dtype=np.float64)
        self.w2 = rng.normal(0, np.sqrt(2.0 / 48), (48, 24))
        self.b2 = np.zeros(24, dtype=np.float64)
        self.w3 = rng.normal(0, np.sqrt(1.0 / 24), (24, output_size))
        self.b3 = np.zeros(output_size, dtype=np.float64)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with non-negative projection."""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        h1 = np.maximum(0.0, arr @ self.w1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.w2 + self.b2)
        out = h2 @ self.w3 + self.b3
        return np.maximum(0.0, out)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 45,
        batch_size: int = 128,
        learning_rate: float = 0.002,
        seed: int = 19,
    ) -> list[float]:
        """Train weights using mini-batch Adam and analytical backpropagation."""
        params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        moments = [np.zeros_like(p) for p in params]
        variances = [np.zeros_like(p) for p in params]
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        rng = np.random.default_rng(seed)
        step = 0
        loss_history = []

        n_samples = len(x)
        for epoch in range(epochs):
            order = rng.permutation(n_samples)
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n_samples, batch_size):
                idx = order[start : start + batch_size]
                xb, yb = x[idx], y[idx]
                
                # Forward pass
                z1 = xb @ self.w1 + self.b1
                h1 = np.maximum(0.0, z1)
                z2 = h1 @ self.w2 + self.b2
                h2 = np.maximum(0.0, z2)
                pred = h2 @ self.w3 + self.b3
                
                # Loss & gradients (MSE)
                batch_err = pred - yb
                epoch_loss += float(np.mean(np.square(batch_err)))
                num_batches += 1
                
                grad = 2.0 * batch_err / len(xb)
                grads = [None] * 6
                grads[4] = h2.T @ grad
                grads[5] = grad.sum(axis=0)
                
                dh2 = grad @ self.w3.T
                dz2 = dh2 * (z2 > 0.0)
                grads[2] = h1.T @ dz2
                grads[3] = dz2.sum(axis=0)
                
                dh1 = dz2 @ self.w2.T
                dz1 = dh1 * (z1 > 0.0)
                grads[0] = xb.T @ dz1
                grads[1] = dz1.sum(axis=0)

                step += 1
                for j, (param, g) in enumerate(zip(params, grads)):
                    moments[j] = beta1 * moments[j] + (1.0 - beta1) * g
                    variances[j] = beta2 * variances[j] + (1.0 - beta2) * (g * g)
                    mhat = moments[j] / (1.0 - beta1**step)
                    vhat = variances[j] / (1.0 - beta2**step)
                    param -= learning_rate * mhat / (np.sqrt(vhat) + epsilon)

            loss_history.append(epoch_loss / max(1, num_batches))

        return loss_history
