# Author:       This code is mostly from TensorFlow tutorial
#               https://www.tensorflow.org/versions/r0.8/tutorials/mnist/beginners/index.html
# Filename:     generate_data.py
# Last edited:  9 May 2016 
# Description:  Simple one layer softmax regression using TensorFlow [1] on
#               NMIST data [2]. Requires TensorFlow installed. Trains a one
#               layer neural network using 100 random image batches of  
#               training data for 1000 iterations. Then it runs the neural
#               network on the first 1000 test images and for each image it 
#               generates a probability vector of being each of 10 possible
#               digits. These probability vectors are what we consider 
#               'features'. We save features and labels on the file 
#               './data/data_features.mat'. (FILENAME can be changed in the
#               first line of code). It is not necessary to run this script
#               since data is already present. This code is present for
#               completeness
#               
#
# Inputs:       
#               
# Outputs:        
# 
# References:
# [1] Abadi et al. TensorFlow: Large-scale machine learning on 
#       heterogeneous systems.
# [2] LeCun, Cortes. Mnist handwritten digit database.
# [3] Mixon, Villar, Ward. Clustering subgaussian mixtures via semidefinite
#       programming
# [4] Peng, Wei. Approximating k-means-type clustering via semidefinite 
#       programming.
# -------------------------------------------------------------------------

FILENAME='./data/data_features.mat'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images[range(1000)], y_: mnist.test.labels[range(1000)]}))

prediction=tf.argmax(y,1)
print "predictions", prediction.eval(feed_dict={x: mnist.test.images}, session=sess)

probabilities=y
pr=probabilities.eval(feed_dict={x: mnist.test.images}, session=sess)


import scipy.io as sio
import numpy as np
sio.savemat(FILENAME, {'digits': np.transpose(pr[range(1000)]), 'labels': np.transpose(mnist.test.labels[range(1000)])} )
