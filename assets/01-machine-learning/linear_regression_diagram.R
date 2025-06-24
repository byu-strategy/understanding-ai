# ------------------------------------------------------------------------------
# Linear Model Geometry Visualization
#
# This script uses ggplot2 to visualize the geometry of a simple linear model:
#     y = w₀ + w₁·x
# It shows:
#   - The regression line defined by the slope (w₁) and intercept (w₀)
#   - The intercept point on the y-axis
#   - A visual "rise over run" triangle to illustrate the slope concept
#   - Annotated math expressions to reinforce the linear equation structure
#
# The resulting plot helps build intuition for how slope and intercept define
# a linear model — useful in teaching machine learning or linear regression.
# ------------------------------------------------------------------------------


# Load ggplot2
library(ggplot2)

# Define weights (ML-style)
w1 <- 1  # slope
w0 <- 2     # intercept (bias)

# Generate data
x_vals <- seq(-5, 5, by = 0.1)
y_vals <- w0 + w1 * x_vals
df <- data.frame(x = x_vals, y = y_vals)

# Triangle: start at intercept (0, w0)
x0 <- 1
y0 <- w0 + 1
x1 <- x0 + 1       # run = 1
y1 <- y0
x2 <- x1
y2 <- y1 + w1      # rise = w1

# Plot
png("images/linear_regression_diagram.png", width = 2400, height = 1800, res = 300)

ggplot(df, aes(x = x, y = y)) +
  geom_line(color = "black", size = 1.2) +
  geom_hline(yintercept = 0, color = "gray50") +
  geom_vline(xintercept = 0, color = "gray50") +
  geom_point(aes(x = 0, y = w0), color = "red", size = 3) +
  
  # Axis limits and 1-unit ticks
  scale_x_continuous(breaks = seq(-5, 5, by = 1), limits = c(-5, 5)) +
  scale_y_continuous(breaks = seq(-5, 5, by = 1), limits = c(-5, 5)) +
  
  # Triangle for slope
  geom_segment(aes(x = x0, y = y0, xend = x1, yend = y1), color = "blue", linetype = "dashed") +  # run
  geom_segment(aes(x = x1, y = y1, xend = x2, yend = y2), color = "blue", linetype = "dashed") +  # rise
  geom_segment(aes(x = x0, y = y0, xend = x2, yend = y2), color = "blue") +    
  
  # Math-style labels using parse = TRUE
  annotate("text", x = -.5, y = w0 + 0, label = "w[0] == Intercept", color = "red", parse = TRUE, hjust = 1) +
  annotate("text", x = 2.5, y = 4, label = "w[1] == Slope", color = "blue", parse = TRUE, hjust = 0) +
  annotate("text", x = -4.8, y = 4.8, label = "y == w[0] + w[1]*x", size = 5, fontface = "bold", parse = TRUE, hjust = 0) +

  labs(x = "x", y = "y") +
  
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank())

dev.off()