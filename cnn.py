# ---------------------------------------------------------------+
# Ver 1.0    Nov 14 2017                                         |
# Author: zxm                                                    |
# Summary: to train a CNN model for fashion MNIST classification |
# ---------------------------------------------------------------+


import tensorflow as tf
import os
import numpy as np
from numpy import genfromtxt
# import matplotlib.pyplot as plt
import time


MAXITER = 5
IMG_ROW = 28
IMG_COL = 28

def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pooling(h):
    return tf.layers.max_pooling2d(h, 2, 2)


# read input data and separate x and y
def read_data(filename):
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    print data.shape
    X = data[:, 1:] / 255.0
    y = data[:, 0]
    print X.shape
    # print y
    print y.shape

    num = int(len(y))
    print num
    Y = np.zeros((num, 10)).astype(float)
    for i in range(num):
        Y[i][int(y[i])] = 1.0

    return X, Y


feature_chnl = 1

feature_layer_1 = 32
feature_layer_2 = 64
feature_layer_3 = 128

feature_layer_soft = 128
feature_layer_final = 10

sess = tf.InteractiveSession()

# paras
W_conv1 = weight_varible([3, 3, feature_chnl, feature_layer_1])
b_conv1 = bias_variable([feature_layer_1])

# conv layer-1
x = tf.placeholder(tf.float32, [None, IMG_ROW * IMG_COL * feature_chnl])
x_image = tf.reshape(x, [-1, IMG_ROW, IMG_COL, feature_chnl])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_conv1 = max_pooling(h_conv1)

# conv layer-2
W_conv2 = weight_varible([3, 3, feature_layer_1, feature_layer_2])
b_conv2 = bias_variable([feature_layer_2])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_conv2 = max_pooling(h_conv2)

# conv layer-3
W_conv3 = weight_varible([3, 3, feature_layer_2, feature_layer_3])
b_conv3 = bias_variable([feature_layer_3])

h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
h_conv3 = max_pooling(h_conv3)

# full connection
W_fc1 = weight_varible([3 * 3 * feature_layer_3, feature_layer_soft])
b_fc1 = bias_variable([feature_layer_soft])

h_conv3_flat = tf.reshape(h_conv3, [-1, 3 * 3 * feature_layer_3])
h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

# output layer: softmax
W_fc2 = weight_varible([feature_layer_soft, feature_layer_final])
b_fc2 = bias_variable([feature_layer_final])

y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, feature_layer_final])

# model training
cross_entropy = - tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()
# create model saver
saver = tf.train.Saver()


start_time = time.time()
train_file = "fashion-mnist_train.csv"
test_file = "fashion-mnist_test.csv"

# train_dataset = open(train_file, 'r')

# print "reading testing data..."
# features_test, labels_test = read_data(test_file)

print "reading training data..."
features, labels = read_data(train_file)
row, col = features.shape

batch_size = 100

save_path = "models/"
if not os.path.exists(save_path): 
    os.makedirs(save_path)

print "training model..."
sess.run(init_op)

for iter in range(MAXITER):
    batch_no = int(row / batch_size)
    # generate batches randomly
    indexes = np.random.permutation(row)

    for i in range(batch_no):
        ind_l = i * batch_size
        ind_r = ind_l + batch_size
        feature = features[indexes[ind_l:ind_r]]
        label = labels[indexes[ind_l:ind_r]]

        train_step.run(feed_dict = {x: feature, y_: label})

        # saver.save(sess, os.path.join(save_path, 'my-model'), global_step = iter)
    train_accuacy = accuracy.eval(feed_dict={x: feature, y_: label})
    print("loop %d/50, training accuracy %g"%(iter+1, train_accuacy)) 

print "reading testing data..."
features_test, labels_test = read_data(test_file)

test_accuacy = accuracy.eval(feed_dict={x: features_test, y_: labels_test})
print("loop %d, testing accuracy %g"%(iter+1, test_accuacy))

print "--- %s seconds ---" % (time.time() - start_time)
