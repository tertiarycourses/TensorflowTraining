# Module 9 Keras
# Challenge: CNN Model on CIFAR-10 dataaset

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
n_classes = 10
learning_rate = 1
training_epochs = 1

# Step 1: Pre-process the data
from tensorflow.python.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

# Step 2: Create the Model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32,32,3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

# Step 3: Compile the Model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

# Step 4: Training
model.fit(X_train, y_train,
          epochs=training_epochs,
          validation_data=[X_test,y_test])

# Step 5: Evaluation
score = model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])