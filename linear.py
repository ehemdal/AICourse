# linear.py: A simple linear regression model using TensorFlow and Keras
# This script creates a model to learn the relationship y = 2x - 1.
# It trains the model on sample data and makes a prediction.
# This code is adapted from a sample provided for 
# AI and Machine Learning for Coders, published by O'Reilly Media, Inc.
# Copyright 2024 by Laurence Moroney.

# Import TensorFlow, NumPy, and Keras modules
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense 

# This model uses one layer with one neuron
# It uses Stochastic Gradient Descent (SGD) as the optimizer
# and Mean Squared Error (MSE) as the loss function.
l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Here is the training data.  We know that y = 2x - 1, but the model doesn't,
# and we're not telling it.  The model has to learn that relationship on its own.
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train the model for 500 steps
model.fit(xs, ys, epochs=500)

# Make a prediction for x = 10.0.  Notice that we have to specify this as a NumPy array.
print(model.predict(np.array([10.0])))

# Display the weights learned by the model
for index, layer in enumerate(model.layers):
    print("Layer {}: {}".format(index, model.get_weights())) 
