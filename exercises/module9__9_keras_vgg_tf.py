# Module 9 Keras
# VGG16 Transfer Learning

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input,decode_predictions
from tensorflow.python.keras.preprocessing import image
import numpy as np

# Step 1: Preprocess data
# Load the image file, resizing it to 224x224 pixels (required by this model)
img = image.load_img("images/chicken.jpg", target_size=(224, 224))

# Convert the image to a numpy array
x = image.img_to_array(img)

# Add a forth dimension since Keras expects a list of images
x = np.expand_dims(x, axis=0)

# Step 2: Load Pre-trained Model
model = VGG16()
print(model.summary())

# Run the image through the deep neural network to make a prediction
predictions = model.predict(x)

# Look up the names of the predicted classes. Index zero is the results for the first image.
predicted_classes = decode_predictions(predictions, top=3)

print("This is an image of:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood*100))

