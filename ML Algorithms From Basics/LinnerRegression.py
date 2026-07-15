import numpy as np
import matplotlib.pyplot as pt

# Data
x = np.array([5, 8, 9, 12, 1, 7, 16, 20, 25])
y = np.array([18, 28, 29, 88, 11, 76, 116, 200, 250])

# Number of data points
n = len(x)

# Summations
sum_xy = np.sum(x * y)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_x_sq = np.sum(x**2)

# Correct Slope (m)
m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - (sum_x**2))

# Correct Intercept (c)
c = (sum_y - m * sum_x) / n

y_predt = m * x + c

## Output m and c
print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")

## Plot the actual data and regression line
pt.scatter(x, y, label='Actual Data')
pt.plot(x, y_predt, color='red', label='Regression Line')
pt.xlabel("x")
pt.ylabel('y')
pt.legend()
pt.show()
