# Module 4: Machine Learning using Tensorflow
# Use softmax cross-entropy function on MINST dataset

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
learning_rate = 0.5
batch_size = 100
tf.set_random_seed(25)

# Step 1: Initial Setup
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.truncated_normal([784, 10],stddev=0.1))
b = tf.Variable(tf.truncated_normal([10],stddev=0.1))

# Step 2: Setup Model
yhat = tf.matmul(X, W) + b

# Step 3: Loss Functions
loss = tf.reduce_mean(
   tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))

# Step 4: Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# % of correct answer found in batches
is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Step 5: Training Loop
for i in range(1000):
    batch_X, batch_y = mnist.train.next_batch(batch_size)
    train_data = {X: batch_X, y: batch_y}
    sess.run(train, feed_dict=train_data)

    print(i+1, "Training accuracy =",sess.run(accuracy, feed_dict=train_data),
          "Loss =",sess.run(loss, feed_dict=train_data))

# Step 6: Evaluation
test_data = {X:mnist.test.images,y:mnist.test.labels}
print("Testing accuracy = ",sess.run(accuracy, feed_dict=test_data))
