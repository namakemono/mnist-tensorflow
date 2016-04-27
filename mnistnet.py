#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def conv(x, n):
    W, b = weight_variable([3,3,channels(x),n]), bias_variable([n])
    return conv2d(x, W) + b

def dense(x, n):
    W, b = weight_variable([volume(x), n]), bias_variable([n])
    return tf.matmul(x, W) + b

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def volume(x):
    return np.prod([d for d in x.get_shape()[1:].as_list()])

def flatten(x):
    return tf.reshape(x, [-1, volume(x)])

def channels(x):
    return int(x.get_shape()[-1])

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def inference(images, keep_prob):
    h = max_pool(tf.nn.relu(conv(images, 32)))
    h = max_pool(tf.nn.relu(conv(h, 64)))
    h = tf.nn.relu(dense(flatten(h), 1024))
    h = tf.nn.dropout(h, keep_prob)
    y = tf.nn.softmax(dense(h, 10))
    return y

def loss(logits, labels):
    return -tf.reduce_mean(labels * tf.log(logits)) # H = -Σ{y' * log(y) + (1-y') * log(1-y)}

def train(total_loss):
    return tf.train.AdamOptimizer(1e-4).minimize(total_loss)
