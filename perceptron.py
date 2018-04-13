import numpy as np
# from numpy import genfromtxt
# import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from keras.datasets import fashion_mnist
import time
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

MAX_PASS = 500

start_time = time.time()
print "loading data..."
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = np.array(x_train).astype(float)
x_test = np.array(x_test).astype(float)
x_train /= 255.0
x_test /= 255.0

mean = np.mean(x_train)
x_train -= mean
x_test -= mean

train_num, row, col = x_train.shape
test_num = x_test.shape[0]

x_train = x_train.reshape((train_num, row*col))
x_test = x_test.reshape((test_num, row*col))

print "training..."
penalty = None
# print "penalty: " + penalty
time1 = time.time()
reg = Perceptron(penalty=penalty, alpha=0.0001, fit_intercept=True, max_iter=MAX_PASS, tol=None, shuffle=True)
reg.fit(x_train, y_train)
print "--- %s seconds ---" % (time.time() - time1)


coef = reg.coef_
row, col = coef.shape
print "sparsity: " + str(1.0 - np.count_nonzero(coef)/float(row*col))
print "testing..."
time2 = time.time()
print "accuracy: " +  str(reg.score(x_test, y_test))
print "--- %s seconds ---" % (time.time() - time2)
print "--- %s seconds ---" % (time.time() - start_time)
