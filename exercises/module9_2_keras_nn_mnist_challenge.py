# Module 9 Keras
# NN Model on MNIST dataset Challenge

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Hyper Parameters
n_features = 784
n_classes = 10
learning_rate = 0.5
training_epochs = 2

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

# Step 2: Build the  Network
L1 = 1024
L2 = 512
L3 = 256
L4 = 128
L5 = 64
L6 = 32

model = Sequential()
model.add(Dense(L1, input_dim=n_features, activation='relu'))
model.add(Dense(L2, activation='relu'))
model.add(Dense(L3, activation='relu'))
model.add(Dense(L4, activation='relu'))
model.add(Dense(L5, activation='relu'))
model.add(Dense(L6, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

print(model.summary())

# Step 3: Compile the Model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Step 4: Train Model
model.fit(X_train, y_train,
          epochs=training_epochs,
          validation_data=(X_test, y_test),
          shuffle=True)

# Step 5: Evaluation
score = model.evaluate(X_test, y_test)
print("\nTraining Accuracy = ",score[1],"Loss",score[0])

