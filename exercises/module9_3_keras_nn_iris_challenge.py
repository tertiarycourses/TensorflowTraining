# Module 9 Keras
# Challenge: NN on Iris dataset

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
n_features = 4
n_classes = 3
learning_rate = 0.08
training_epochs = 100

# Step 1: Preprocess the  Data
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
target = iris.target

# Convert the label into one-hot vector
num_labels = len(np.unique(target))
Y = np.eye(num_labels)[target]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Step 2: Build the  Network
L1 = 100
L2 = 40
L3 = 20
model = Sequential()
model.add(Dense(L1, input_dim=n_features, activation='relu'))
model.add(Dense(L2, activation='relu'))
model.add(Dense(L3, activation='relu'))
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

# Step 6: Save the model
model.save("./models/iris.h5")