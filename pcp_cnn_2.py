# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 14:07:56 2018

@author: jaimeHP
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import pcp_data_cnn_2
import matplotlib.pyplot as plt
import random as ran

tf.logging.set_verbosity(tf.logging.INFO)

GROUP = ['SAL', 'PCP']

def TRAIN_SIZE(num):
    print ('Total Training Images in Dataset = ' + str(mnist.train.images.shape))
    print ('--------------------------------------------------')
    x_train = mnist.train.images[:num,:]
    print ('x_train Examples Loaded = ' + str(x_train.shape))
    y_train = mnist.train.labels[:num,:]
    print ('y_train Examples Loaded = ' + str(y_train.shape))
    print('')
    return x_train, y_train

def TEST_SIZE(num):
    print ('Total Test Examples in Dataset = ' + str(mnist.test.images.shape))
    print ('--------------------------------------------------')
    x_test = mnist.test.images[:num,:]
    print ('x_test Examples Loaded = ' + str(x_test.shape))
    y_test = mnist.test.labels[:num,:]
    print ('y_test Examples Loaded = ' + str(y_test.shape))
    return x_test, y_test

def display_digit(num):
    print(y_train[num])
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28,28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def display_mult_flat(start, stop):
    images = x_train[start].reshape([1,784])
    for i in range(start+1,stop):
        images = np.concatenate((images, x_train[i].reshape([1,784])))
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()

#def cnn_model_fn(features, labels, mode):
#  """Model function for CNN."""
#  # Input Layer
#  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
#
#  # Convolutional Layer #1
#  conv1 = tf.layers.conv2d(
#      inputs=input_layer,
#      filters=32,
#      kernel_size=[5, 5],
#      padding="same",
#      activation=tf.nn.relu)
#
#  # Pooling Layer #1
#  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
#
#  # Convolutional Layer #2 and Pooling Layer #2
#  conv2 = tf.layers.conv2d(
#      inputs=pool1,
#      filters=64,
#      kernel_size=[5, 5],
#      padding="same",
#      activation=tf.nn.relu)
#  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#
#  # Dense Layer
#  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
##  print(len(pool2_flat))
#  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
#  dropout = tf.layers.dropout(
#      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
#
#  # Logits Layer
#  logits = tf.layers.dense(inputs=dropout, units=10)
#  
#
#  predictions = {
#      # Generate predictions (for PREDICT and EVAL mode)
#      "classes": tf.argmax(input=logits, axis=1),
#      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
#      # `logging_hook`.
#      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
#  }
#
#  if mode == tf.estimator.ModeKeys.PREDICT:
#    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
#
#  # Calculate Loss (for both TRAIN and EVAL modes)
#  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
#  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
#
#  # Configure the Training Op (for TRAIN mode)
#  if mode == tf.estimator.ModeKeys.TRAIN:
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#    train_op = optimizer.minimize(
#        loss=loss,
#        global_step=tf.train.get_global_step())
#    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
#
#  # Add evaluation metrics (for EVAL mode)
#  eval_metric_ops = {
#      "accuracy": tf.metrics.accuracy(
#          labels=labels, predictions=predictions["classes"])}
#  return tf.estimator.EstimatorSpec(
#      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
#    Load training and eval data
    x_train, y_train = TRAIN_SIZE(55000)
    
#    (train_data, train_labels), (eval_data, eval_labels) = pcp_data_cnn.load_data(datamode='limited')
#    
#    print(train_labels.dtype)
#    print(np.shape(train_labels))
#      # Create the Estimator
#    mnist_classifier = tf.estimator.Estimator(
#        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
#    
## Set up logging for predictions
#    tensors_to_log = {"probabilities": "softmax_tensor"}
#    logging_hook = tf.train.LoggingTensorHook(
#        tensors=tensors_to_log, every_n_iter=50)
#    
#    # Train the model
#    train_input_fn = tf.estimator.inputs.numpy_input_fn(
#        x={"x": train_data},
#        y=train_labels,
#        batch_size=100,
#        num_epochs=None,
#        shuffle=True)
#    mnist_classifier.train(
#        input_fn=train_input_fn,
#        steps=10000,
#        hooks=[logging_hook])
#
## Evaluate the model and print results
#    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#        x={"x": eval_data},
#        y=eval_labels,
#        num_epochs=1,
#        shuffle=False)
#    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
#    print(eval_results)

if __name__ == "__main__":
  tf.app.run()

