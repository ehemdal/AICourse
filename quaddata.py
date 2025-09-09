import numpy as np
import random
import matplotlib.pyplot as plot

# Coefficients for y = ax^2 + bx + c
a = 3
b = 2
c = 1

# Generate evenly spaced x values from -10 to 10 into a NumPy array
x = np.linspace(-10, 10, 100)

# Generate training y values using the quadratic equation also into a NumPy array.
y = a * x**2 + b * x + c #+ 0.1*np.random.normal(size=1000)
print(x)
print(y)
# Save the data to a file
np.savez('quadratic_train_data.npz', x=x, y=y)

# Generate noisy test data
y = a * x**2 + b * x + c + 0.2*np.random.normal(size=100)
print(x)
print(y)
# Save the data to a file
np.savez('quadratic_test_data.npz', x=x, y=y)

# plot the training data
plot.scatter(x, y)
plot.grid()
plot.show()
plot.savefig('quadratic_train_data.png')

# Plot the test data
plot.scatter(x, y)  
plot.grid()
plot.show()
plot.savefig('quadratic_test_data.png')

# Load the data back from the file to verify it was saved correctly
data = np.load('quadratic_train_data.npz')
print(data['x'])
print(data['y'])

