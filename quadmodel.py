# quadmodel.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Load the data back from the file to verify it was saved correctly
data = np.load('quadratic_data.npz')
xs = data['x']
ys = data['y']  

# Define the model
model = Sequential([
    Dense(units=1,activation='linear',input_shape=[1]),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=1,activation='linear')
])      
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()
# Train the model
model.fit(xs, ys, epochs=1000, verbose=0) 
# Make a prediction for x = 10.0
print(model.predict(np.array([10.0])))  
# Display the weights learned by the model
for index, layer in enumerate(model.layers):
    print("Layer {}: {}".format(index, model.get_weights()))    
