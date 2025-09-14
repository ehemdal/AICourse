import numpy as np
import random
import matplotlib.pyplot as plot

# Coefficients for y = ax^2 + bx + c
a = 3
b = 2
c = 1

# Generate evenly spaced x values from PI to PI into a NumPy array
x_train = np.linspace(-3.14, 3.14, 100)

# Generate ideal y values into a NumPy array.
y_ideal = np.sin(x_train)
print(x_train)
print(y_ideal)
# Save the data to a file
# np.savez('quadratic_train_data.npz', x=x_train, y=y_train)

# Generate noisy training data
y_train = np.sin(x_train) + 0.2*np.random.normal(size=100)
print(x_train)
print(y_train)
# Save the data to a file
np.savez('sine_training_data.npz', x=x_train, y=y_train)

# plot the training data
plot.scatter(x_train, y_train)
plot.scatter(x_train, y_ideal, color='red')
plot.title('Noisy Training Data and Ideal Data')
plot.grid()
plot.show()
plot.savefig('sine_train_data.png')

