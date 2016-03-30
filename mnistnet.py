#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def inference(images, keep_prob):
    # Build a Multilayer Convolutional Network: INPUT -> [CONV -> RELU -> POOL] * 2 -> FC -> RELU -> FC
    W1, b1 = weight_variable([3,3,1,32]), bias_variable([32]) # 3x3 Filter, input channel: 1, output channel: 32
    W2, b2 = weight_variable([3,3,32,64]), bias_variable([64]) # 3x3 Filter, input channel: 32, output channel: 64
    W3, b3 = weight_variable([7*7*64,1024]), bias_variable([1024])
    W4, b4 = weight_variable([1024,10]), bias_variable([10])
    x = tf.reshape(images, [-1, 28, 28, 1]) # 28x28, channel=1
    h1 = max_pool_2x2(tf.nn.relu(conv2d(x, W1) + b1)) # First Convolutional Layer: CONV -> RELU -> POOL, image size: 28x28 -> 14x14
    h2 = max_pool_2x2(tf.nn.relu(conv2d(h1, W2) + b2)) # Second Convolutional Layer: CONV -> RELU -> POOL, image size: 14x14 -> 7x7
    h3 = tf.nn.relu(tf.matmul(tf.reshape(h2, [-1, 7*7*64]), W3) + b3) # Densely Connected Layer: FC -> RELU
    h4 = tf.nn.dropout(h3, keep_prob) # Dropout
    y = tf.nn.softmax(tf.matmul(h4, W4) + b4) # Readout Layer: FC 
    return y

def loss(logits, labels):
    return -tf.reduce_sum(labels * tf.log(logits)) # H = -Σ{y' * log(y) + (1-y') * log(1-y)}

def train(total_loss):
    return tf.train.AdamOptimizer(1e-4).minimize(total_loss)
