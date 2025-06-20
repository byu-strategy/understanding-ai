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
plt.show()



##### MSE

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(42)
X = np.random.uniform(1200, 3000, size=(30, 1))
y = 50000 + 200 * X + np.random.normal(0, 30000, size=(30, 1))

# Fit a linear model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plot data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Observed Data")
plt.plot(X, y_pred, color="red", label="Predicted Line (Model Output)")

# Plot error lines
for i in range(len(X)):
    plt.plot([X[i], X[i]], [y[i], y_pred[i]], color="gray", linestyle="--", linewidth=0.8)

# Add dummy line for legend entry
plt.plot([], [], 'k--', label="Loss (Prediction error for each example)")

plt.title("Visualizing Loss: What the Loss Function Measures")
plt.xlabel("Input Feature (e.g., Square Footage)")
plt.ylabel("Output (e.g., Home Price)")
plt.legend()
plt.grid(True)
plt.show()


## Loss function plot

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample training data
X = np.array([1200, 1400, 1600])
y = np.array([250000, 275000, 300000])

# Create design matrix for solving optimal weights
X_design = np.column_stack((np.ones_like(X), X))
w_opt = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
w0_opt, w1_opt = w_opt
loss_opt = np.mean((w0_opt + w1_opt * X - y) ** 2)

# Wider grid to see full bowl
w0_range = np.linspace(-300000, 300000, 100)
w1_range = np.linspace(0, 500, 100)
W0, W1 = np.meshgrid(w0_range, w1_range)

# Compute loss surface
Z = np.zeros_like(W0)
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        w0 = W0[i, j]
        w1 = W1[i, j]
        y_pred = w0 + w1 * X
        Z[i, j] = np.mean((y_pred - y) ** 2)

# Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W0, W1, Z, cmap='viridis', alpha=0.9)

# Add optimal point in red
ax.scatter(w0_opt, w1_opt, loss_opt, color='red', s=60, label='Optimal (w₀, w₁)')
ax.legend()

# Labels and title
ax.set_xlabel(r'$w_0$ (Intercept)')
ax.set_ylabel(r'$w_1$ (Slope)')
ax.set_zlabel(r'$\mathcal{L}(w_0, w_1)$ (MSE Loss)')
ax.set_title("Full Bowl-Shaped Loss Surface for Linear Regression")

plt.tight_layout()
plt.show()



