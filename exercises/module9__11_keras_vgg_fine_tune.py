# Module 9 Keras
# VGG16 Fine Tuning

from tensorflow.python import keras
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Input

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.python.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test,10)

# Step 1: Create the base pre-trained model
input_tensor = Input(shape=(224, 224, 3))
base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

# Step 2: Create a new model with dense and softamx layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(17, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Step 3: Freeze all pre-trained layers and train the top layers with new dataaset
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=100, epochs=2)

# Step 4: Unfreeze some pre-trained layers and train with new dataset
for layer in model.layers[:5]:
    layer.trainable = False
for layer in model.layers[5:]:
    layer.trainable = True

from tensorflow.python.keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
model.fit(X_train, y_train, batch_size=100, epochs=2)