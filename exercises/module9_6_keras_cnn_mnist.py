# Module 9 Keras
# CNN Model on MNIST dataaset

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
n_classes = 10
learning_rate = 0.5
training_epochs = 2
batch_size = 100

# Step 1 Load the Data
from tensorflow.python.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

# Step 2: Build the Network
model = Sequential()
model.add(Conv2D(32,(3, 3),activation='relu',input_shape=(28,28,1),padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64,(3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
print(model.summary())

# Step 3: Compile the Model
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

# Step 4: Training
model.fit(X_train, y_train,
          epochs=training_epochs)

# Step 4: Evaluation
score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])