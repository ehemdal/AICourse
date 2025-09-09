# quadmodel.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plot


# Load the data back from the file to verify it was saved correctly
data = np.load('quadratic_train_data.npz')
xs = data['x']
ys = data['y']  

# Define the model
model = Sequential([
    Dense(units=1,activation='linear',input_shape=[1]),
    Dense(units=80, activation='relu'),
    Dense(units=80, activation='relu'),
    Dense(units=1,activation='linear')
])      
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()
# Train the model
model.fit(xs, ys, epochs=250, verbose=0) 
# Make a prediction for x = 5.0 should be 86
print(model.predict(np.array([5.0])))  

# Try the model using test data
testdata = np.load('quadratic_test_data.npz')
test_xs = testdata['x']
test_ys = testdata['y']
test_loss, test_mae = model.evaluate(test_xs, test_ys, verbose=2)
print('\nTest MAE:', test_mae)
test_ys = model.predict(test_xs)
# Plot the test data
plot.scatter(test_xs, test_ys)
plot.grid()
plot.show()
plot.savefig('quadratic_test_results.png')  

# Display the weights learned by the model
# for index, layer in enumerate(model.layers):
#     print("Layer {}: {}".format(index, model.get_weights()))    
