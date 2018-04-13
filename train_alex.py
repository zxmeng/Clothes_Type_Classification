import numpy as np
from numpy import genfromtxt
from PIL import Image
import keras
from keras.optimizers import Adam
from keras.models import load_model
import alexnet
import time

# MODEL = 'my_model_001.h5'
# NEWMODEL = 'my_model_002.h5'

def scale(x_train, x_test):
	x_train = x_train.astype(float)
	x_test = x_test.astype(float)

	x_train /= 255.0
	x_test /= 255.0

	mean = np.mean(x_train)
	x_train -= mean
	x_test -= mean

	return x_train, x_test


start_time = time.time()
# ========== process image data into csv file ==========
# x_train = np.zeros((71093, 1024)).astype(int)
# y_train = np.zeros((71093)).astype(int)

# x_test = np.zeros((17858, 1024)).astype(int)
# y_test = np.zeros((17858)).astype(int)

# print "loading train data..."
# traintxt = open("fashion-data/train.txt", 'r')
# i = 0
# for line in traintxt:
# 	line = line.strip("\n")
# 	line = line.replace("/", "_")
# 	filename = "processed/" + line + ".jpg"
# 	img = Image.open(filename)
# 	x_train[i] = img.getdata()
# 	token = line.split("_")
# 	y_train[i] = int(token[0])
# 	i += 1
# print i

# print "loading test data..."
# testtxt = open("fashion-data/test.txt", 'r')
# i = 0
# for line in testtxt:
# 	line = line.strip("\n")
# 	line = line.replace("/", "_")
# 	filename = "processed/" + line + ".jpg"
# 	img = Image.open(filename)
# 	x_test[i] = img.getdata()
# 	token = line.split("_")
# 	y_test[i] = int(token[0])
# 	i += 1
# print i

# print "saving data..."
# np.savetxt('x_train.csv', x_train, fmt='%d', delimiter=',')
# np.savetxt('y_train.csv', y_train, fmt='%d', delimiter=',')
# np.savetxt('x_test.csv', x_test, fmt='%d', delimiter=',')
# np.savetxt('y_test.csv', y_test, fmt='%d', delimiter=',')
# ========== process image data into csv file ==========

print "loading data..."
x_train = genfromtxt("x_train.csv", delimiter=',')
y_train = genfromtxt("y_train.csv", delimiter=',')

x_test = genfromtxt("x_test.csv", delimiter=',')
y_test = genfromtxt("y_test.csv", delimiter=',')

x_train, x_test = scale(x_train, x_test)
x_train = x_train.reshape((71093, 32, 32, 1))
x_test = x_test.reshape((17858, 32, 32, 1))
y_train = keras.utils.to_categorical(y_train, 15)
y_test = keras.utils.to_categorical(y_test, 15)

print "training..."
# build vgg11 model
# input_shape = (32, 32, 1)
# model = alexnet.build_model(input_shape)

MODEL= "alex_acs_009.h5"
model = load_model(MODEL)

# training and testing
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
for i in range(10):
	history = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test))
	# save trained model for future usage
	model.save("alex_acs_01" + str(i) + ".h5")

# evaluate model
score = model.evaluate(x_test, y_test)
print('test loss: ', score[0])
print('test accuracy: ', score[1])

print "--- %s seconds ---" % (time.time() - start_time)


