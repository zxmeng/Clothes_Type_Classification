import numpy as np
from keras.datasets import fashion_mnist

def read_data():
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

	return x_train, y_train, x_test, y_test
