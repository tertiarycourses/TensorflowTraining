# Module 9 Keras
# CNN Model on MNIST dataaset

from tensorflow.python import keras
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Hyper Parameters
n_features = 784
n_classes = 10

# Step 1: Pre-process the  Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

model = load_model('mnist.h5')

X = X_test[0].reshape([-1,784])
prediction = model.predict(X)
print(prediction.argmax(axis=1))

def show_digit(index):
    label = y_test[index].argmax(axis=0)
    image = X_test[index].reshape([28,28])
    plt.title('Digit : {}'.format(label))
    plt.imshow(image, cmap='gray_r')
    plt.show()

show_digit(0)