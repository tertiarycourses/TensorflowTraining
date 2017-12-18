# Module 9 Keras
# Challenge: NN on Iris dataset

from tensorflow.python.keras.models import load_model
import numpy as np

X = [3.1,2.1,4.1,5.5]
X = np.reshape(X,[-1,4])

flower = {0:"sentosa",1:"vicolor",2:"virgica"}

model = load_model('iris_100.h5')

prediction = model.predict(X)
prediction = np.argmax(prediction)

print("This flower is ", flower[prediction])



