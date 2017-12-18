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
from tensorflow.python.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

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