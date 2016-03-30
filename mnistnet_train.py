#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import input_data
import mnistnet

def main(args):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Session() as sess:
        x = tf.placeholder("float", shape=[None, 784])
        t = tf.placeholder("float", shape=[None, 10])
        keep_prob = tf.placeholder("float")
        is_train = tf.placeholder("bool")
        y = mnistnet.inference(x, keep_prob)
        cross_entropy = mnistnet.loss(y, t)
        if is_train:
            optimizer = mnistnet.train(cross_entropy)
        sess.run(tf.initialize_all_variables())
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        for i in range(1000):
            images, labels = mnist.train.next_batch(50)
            _ = sess.run(fetches=optimizer, feed_dict={x: images, t: labels, keep_prob: 0.5, is_train: True})
            if i % 100 == 0:
                train_acc = accuracy.eval(feed_dict={x: images, t: labels, keep_prob: 1.0, is_train: False})
                test_acc = accuracy.eval(feed_dict={x: mnist.test.images, t: mnist.test.labels, keep_prob: 1.0, is_train: False})
                print "[%d]\ttrain-accuracy:%.5f\ttest-accuracy:%.5f" % (i, train_acc, test_acc)
        print "Accuracy: %.3f" % accuracy.eval(feed_dict={x: mnist.test.images, t: mnist.test.labels, keep_prob: 1.0, is_train: False})

if __name__ == "__main__":
    tf.app.run()

