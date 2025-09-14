# quadmodel.py
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plot


# Load the training data
data = np.load('sine_training_data.npz')
xs = data['x']
ys = data['y']  

# Define the model
model = Sequential([
    Dense(units=1, activation='linear', input_shape=([1])),
    Dense(units=1,activation='linear', input_shape=[1]),
    Dense(units=80, activation='relu'),
    Dense(units=80, activation='relu'),
    Dense(units=1,activation='linear')
])      
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()
# Train the model
model.fit(xs, ys, epochs=250, verbose=0) 
# Make a prediction 
result = model.predict(xs)
# print(model.predict(np.array(xs)))  


test_loss, test_mae = model.evaluate(xs, result, verbose=2)
print('\nTest MAE:', test_mae)

# Plot the results
plot.scatter(xs, ys)
plot.title("Test Data with Model Predictions")
plot.scatter(xs, result, color='red')  
plot.grid()
plot.show(block=True)
plot.savefig('sine_test_results.png')  

# Display the weights learned by the model
for index, layer in enumerate(model.layers):
    print("Layer {}: {}".format(index, model.get_weights()))    
