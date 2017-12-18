# Module 4: Simple TF Model
# Simple TF Model - Linear Regression

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

# Step 1: Initial Setup
X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
W = tf.Variable([0.1],tf.float32)
b = tf.Variable([0.1],tf.float32)

X_train = [1,2,3,4,5]
y_train = [0,-1.5,-1.6,-3.1,-4]

# X_train = [1.0,2.0,3.0,4.0,5.0]
# y_train = [4.5,7.9,10.3,10.8,14.5]

# Step 2: Model
yhat = tf.multiply(W,X) + b

# # Step 3: Loss Function
loss = tf.reduce_sum(tf.square(yhat - y)) # sum of the squares error

# # Step 4: Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# # training data
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# # Step 5: Training Loop
for i in range(1000):
  sess.run(train, {X:X_train, y:y_train})

# Step 6: Evaluation
print(sess.run(W))
print(sess.run(b))
import matplotlib.pyplot as plt
plt.plot(X_train,y_train,'o')
plt.plot(X_train,sess.run(tf.multiply(W,X_train)+b),'r')
plt.show()

