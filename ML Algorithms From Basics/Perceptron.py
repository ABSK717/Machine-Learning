import numpy as np

# -----------------------------
# Training Data (AND Gate)
# -----------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# -----------------------------
# Perceptron Parameters
# -----------------------------
learning_rate = 0.1
epochs = 10

# Initialize weights and bias
weights = np.zeros(X.shape[1])
bias = 0

# Step Activation Function
def step(x):
    return 1 if x >= 0 else 0

# -----------------------------
# Training
# -----------------------------
for epoch in range(epochs):

    print(f"\nEpoch {epoch+1}")

    for i in range(len(X)):

        # Weighted Sum
        z = np.dot(X[i], weights) + bias

        # Prediction
        prediction = step(z)

        # Error
        error = y[i] - prediction

        # Update Rule
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

        print(f"Input: {X[i]}  "
              f"Prediction: {prediction}  "
              f"Target: {y[i]}  "
              f"Error: {error}")

    print("Weights:", weights)
    print("Bias:", bias)

# -----------------------------
# Testing
# -----------------------------
print("\nFinal Predictions")

for x in X:
    z = np.dot(x, weights) + bias
    pred = step(z)
    print(f"{x} --> {pred}")
