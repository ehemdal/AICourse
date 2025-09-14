# Adapted from fashion-mnist.py from "AI and Machine Learning for Coders" by Laurence Moroney.
# https://www.apache.org/licenses/LICENSE-2.0

import tensorflow as tf
# The fashion dataset is built into Keras and contains 70,000 grayscale images in 10 categories.
# The images show individual articles of clothing at low resolution (28 by 28 pixels).  
# The training set has 60,000 images, and the test set has 10,000 images.

# Keras provides a helper method to load the dataset and to split it into a training and test set:
data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()
# The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. Normalize.
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Here's the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))     
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=20)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[500])
print(test_labels[500])