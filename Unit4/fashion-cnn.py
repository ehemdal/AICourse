# Adapted from fashion-cnn.py from AI and Machine Learning for Coders by Laurence Moroney.
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf
data = tf.keras.datasets.fashion_mnist

# Define a callback to stop training once we reach 99% accuracy.
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

# Register the callback.
callbacks = myCallback()

(training_images, training_labels), (test_images, test_labels) = data.load_data()

print("The training images are of type " + str(type(training_images)))

# Reshape the images to 2D and normalize the pixel values.
training_images=training_images.reshape(60000, 28, 28, 1)
training_images  = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

# Here's the CNN model.  Rather than using the add() method to add layers one at a time,
# this code passes a list of layers to the Sequential() method.
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
# Notice that we flatten the images only after all the convolution and pooling layers.
# The rest of this model is the same as in the previous example.    
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# We are guessing that 50 epochs should be enough to get to 99% accuracy.
# The callback will interrupt training if we reach that goal sooner.
model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

# Feel free to change the index here to see the result for a different image.
classifications = model.predict(test_images)
print(classifications[500])
print(test_labels[500])