# Module 4: Machine Learning using Tensorflow
# Load graph and model

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Step 1: Restore Graph
sess = tf.Session()
saver = tf.train.import_meta_graph('./tmp/mnist.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./tmp'))

# Step 2: Restore Input and Output
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
yhat = graph.get_tensor_by_name("yhat:0")


# Step 3: Evaluation
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

X_test = mnist.test.images[2:3]
y_test = mnist.test.labels[2]

print("Actual answer : ",sess.run(tf.argmax(y_test)))
print("Predicted answer : ",sess.run(tf.argmax(yhat,1), feed_dict={X: X_test}))
