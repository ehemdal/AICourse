# show_mnist.py: Sample script to show random images from the Fashion MNIST dataset.
# Adapted from https://stackoverflow.com/questions/42353676/display-mnist-image-using-matplotlib
# To use this with the original MNIST dataset, change the import line to refer to mnist and update the label decoding.

from matplotlib import pyplot as plot
from tensorflow import keras
import numpy as np
from keras import datasets

def decode_label(label):
    if label == 0:
        return "T-shirt/top"
    elif label == 1:
        return "Trouser"
    elif label == 2:
        return "Pullover"
    elif label == 3:
        return "Dress"
    elif label == 4:
        return "Coat"
    elif label == 5:
        return "Sandal"
    elif label == 6:
        return "Shirt"
    elif label == 7:
        return "Sneaker"
    elif label == 8:
        return "Bag"
    elif label == 9:
        return "Ankle boot"
    else:
        return "Unknown"

(training_images, training_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
def gen_image(img, label):
    two_d_img = (np.reshape(img, (28, 28)) * 255).astype(np.uint8)
    plot.imshow(two_d_img, interpolation='nearest')
    plot.title("This is a " + decode_label(label) + " - " + str(label))
    return plot

# Get a random image and show it in a pop-up window with its label.
indices = np.random.choice(test_images.shape[0], 2, replace=False)
this_label = test_labels[indices[0]]
print("This has label: ", this_label)
gen_image(test_images[indices[0]],test_labels[indices[0]]).show()

