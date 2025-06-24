"""
This script compares linear regression models trained using two different frameworks:
1. scikit-learn (with standardized inputs using StandardScaler)
2. PyTorch (with manual standardization and gradient descent)

It generates synthetic housing data (square footage vs. home value), fits both models, 
rescales their learned parameters back to the original feature scale, and visualizes 
the fitted lines alongside the original data.

The output is a side-by-side plot saved as an image file showing the regression lines 
from both models for visual comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# --- Set Seed for Reproducibility ---
np.random.seed(42)
torch.manual_seed(42)

# --- Randomly Generate Data ---
n_samples = 30
X_data = np.random.uniform(1200, 3000, size=(n_samples, 1)).astype(np.float32)
true_slope = 200
true_intercept = 50000
noise = np.random.normal(0, 30000, size=(n_samples, 1)).astype(np.float32)
y_data = true_intercept + true_slope * X_data + noise

# --- scikit-learn Model ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

lr_model = LinearRegression()
lr_model.fit(X_scaled, y_data)

# Convert scikit-learn model back to original units
slope_sk = lr_model.coef_[0][0] / scaler.scale_[0]
intercept_sk = lr_model.intercept_[0] - slope_sk * scaler.mean_[0]

# --- PyTorch Model ---
X_torch = torch.tensor(X_data)
y_torch = torch.tensor(y_data)

X_mean = X_torch.mean()
X_std = X_torch.std(unbiased=False)
X_scaled_torch = (X_torch - X_mean) / X_std

model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for _ in range(3000):
    y_pred = model(X_scaled_torch)
    loss = loss_fn(y_pred, y_torch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Convert PyTorch model back to original units
slope_torch = model.weight.item() / X_std.item()
intercept_torch = model.bias.item() - slope_torch * X_mean.item()

# --- Plot ---
X_plot = np.linspace(1100, 3100, 100).reshape(-1, 1)
y_sk = intercept_sk + slope_sk * X_plot
y_pt = intercept_torch + slope_torch * X_plot

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Plot scikit-learn
axes[0].scatter(X_data, y_data, label="Actual Data", color="blue")
axes[0].plot(X_plot, y_sk, label="scikit-learn Fit", color="green")
axes[0].set_title("scikit-learn Regression")
axes[0].set_xlabel("Square Footage")
axes[0].set_ylabel("Home Value")
axes[0].legend()
axes[0].grid(True)

# Plot PyTorch
axes[1].scatter(X_data, y_data, label="Actual Data", color="blue")
axes[1].plot(X_plot, y_pt, label="PyTorch Fit", color="black")
axes[1].set_title("PyTorch Regression")
axes[1].set_xlabel("Square Footage")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("sklearn_vs_pytorch_regression.png", dpi=300)  # Save to file
plt.show()
