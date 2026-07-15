import numpy as np
import matplotlib.pyplot as plt

# Data: x (input) and y (output)
# Generate a larger array of x values
x = np.random.uniform(1, 100, size=100)  # 1000 data points, random values between 1 and 100

# Generate corresponding y values based on a linear equation: y = m*x + c + noise
m_true = 2  # True slope
c_true = 5  # True intercept
noise = np.random.normal(0, 10, size=100)  # Adding some random noise to the data

# Calculate y based on the linear model with noise
y = m_true * x + c_true #+ noise

# Function to compute cost (Mean Squared Error)
def compute_cost(x, y, m, c):
    n = len(y)
    predictions = m * x + c
    cost = (1 / (2 * n)) * np.sum((y - predictions) ** 2)
    return cost

# Function to perform gradient descent
def gradient_descent(x, y, m, c, learning_rate, iterations):
    n = len(y)  # Number of data points
    cost_history = []  # To store cost values for each iteration

    for i in range(iterations):
        # Compute predictions
        predictions = m * x + c
        
        # Calculate gradients
        dm = -(1 / n) * np.sum((y - predictions) * x)  # Partial derivative w.r.t m
        dc = -(1 / n) * np.sum(y - predictions)       # Partial derivative w.r.t c
        
        # Update parameters
        m = m - learning_rate * dm
        c = c - learning_rate * dc
        
        # Compute cost and store it
        cost = compute_cost(x, y, m, c)
        cost_history.append(cost)
        
        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}, m = {m:.4f}, c = {c:.4f}")
    
    return m, c, cost_history

# Initialize parameters
m_init = 0     # Initial slope
c_init = 0     # Initial intercept
learning_rate = 0.0005  # Small learning rate
iterations = 1000       # Number of iterations

# Perform gradient descent
m_opt, c_opt, cost_history = gradient_descent(x, y, m_init, c_init, learning_rate, iterations)

# Final results
print(f"\nOptimal parameters: m = {m_opt:.4f}, c = {c_opt:.4f}")

# Plot the results
plt.scatter(x, y,  label='Data Points')       # Actual data
plt.plot(x, m_opt * x + c_opt, color='red', label='Best Fit Line')  # Best-fit line
plt.xlabel("x")
plt.ylabel("y")
plt.title("Gradient Descent - Linear Regression")
plt.legend()
plt.show()

# Plot cost over iterations
plt.plot(range(iterations), cost_history, color='purple')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.show()
