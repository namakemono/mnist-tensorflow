#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import input_data
import mnistnet
import time

def main(args):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Session() as sess:
        x = tf.placeholder("float", shape=[None, 784])
        t = tf.placeholder("float", shape=[None, 10])
        keep_prob = tf.placeholder("float")
        y = mnistnet.inference(tf.reshape(x, [-1, 28, 28, 1]), keep_prob)
        loss = mnistnet.loss(y, t)
        optimizer = mnistnet.train(loss)
        sess.run(tf.initialize_all_variables())
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        batch_size = 50
        for i in range(100):
            start_time = time.time()
            N = 50000
            for j in range(0, N, batch_size):
                images, labels = mnist.train.next_batch(batch_size)
                _ = sess.run(fetches=optimizer, feed_dict={x: images, t: labels, keep_prob: 0.5})
            duration = time.time() - start_time
            examples_per_sec = N / duration
            train_acc = accuracy.eval(feed_dict={x: images, t: labels, keep_prob: 1.0})
            test_acc = accuracy.eval(feed_dict={x: mnist.test.images, t: mnist.test.labels, keep_prob: 1.0})
            print "[%d]\ttrain-accuracy:%.5f\ttest-accuracy:%.5f(%.1f examples/sec)" % (i, train_acc, test_acc, examples_per_sec)
        print "Accuracy: %.3f" % accuracy.eval(feed_dict={x: mnist.test.images, t: mnist.test.labels, keep_prob: 1.0})

if __name__ == "__main__":
    tf.app.run()

