# This plot shows a dataset of input-output pairs along with the best fit regression line.
# Dashed vertical lines represent the prediction error (loss) for each example—
# the distance between the actual value and the predicted value.
# This helps visually illustrate how the loss function measures model performance.


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
plt.plot([], [], 'k--', label="Error (Loss for each example)")

plt.title("Visualizing Prediction Error: What the Loss Function Measures")
plt.xlabel("Input Feature (e.g., Square Footage)")
plt.ylabel("Output (e.g., Home Price)")
plt.legend()
plt.grid(True)

# Save the plot to file
plt.savefig("images/error_visualization_line_fit.png", dpi=300, bbox_inches='tight')

plt.show()
