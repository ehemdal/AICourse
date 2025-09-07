import numpy as np
import random

# Coefficients for y = ax^2 + bx + c
a = 3
b = 2
c = 1

# Generate evenly spaced x values from -10 to 10 into a NumPy array
x = np.linspace(-10, 10, 1000)

# Generate y values using the quadratic equation also into a NumPy array.
y = a * x**2 + b * x + c #+ 0.1*np.random.normal(size=1000)
print(x)
print(y)
# Save the data to a file
np.savez('quadratic_data.npz', x=x, y=y)

# Load the data back from the file to verify it was saved correctly
data = np.load('quadratic_data.npz')
print(data['x'])
print(data['y'])

