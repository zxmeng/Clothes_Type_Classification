import numpy as np
from numpy import genfromtxt
from PIL import Image
import keras
from keras.optimizers import Adam
from keras.models import load_model
import alexnet
import time
import os

# MODEL = 'sub_model_001.h5'
# NEWMODEL = 'my_model_002.h5'

def scale(x_train):
	x_train = x_train.astype(float)
	x_train /= 255.0
	mean = np.mean(x_train)
	x_train -= mean

	return x_train


start_time = time.time()
# ========== process image data into csv file ==========
# x_train = np.zeros((12759, 1024)).astype(int)
# y_train = np.zeros((12759)).astype(int)

# print "loading data..."
# path = "cat_pro/"
# i = 0
# for filename in os.listdir(path):
# 	img = Image.open(path+filename)
# 	x_train[i] = img.getdata()
# 	token = filename.split("_")
# 	y_train[i] = int(token[0])
# 	i += 1
# print i

# print "saving data..."
# np.savetxt('x_sub.csv', x_train, fmt='%d', delimiter=',')
# np.savetxt('y_sub.csv', y_train, fmt='%d', delimiter=',')
# ========== process image data into csv file ==========
# print "--- %s seconds ---" % (time.time() - start_time)

print "loading data..."
x_train = genfromtxt("x_sub.csv", delimiter=',')
y_train = genfromtxt("y_sub.csv", delimiter=',')

x_train = scale(x_train)
x_train = x_train.reshape((12759, 32, 32, 1))
y_train = keras.utils.to_categorical(y_train, 13)

print "training..."
# build alexnet model
input_shape = (32, 32, 1)
# model = alexnet.build_model_d3(input_shape)

MODEL = "alexd3_model_009.h5"
print MODEL
model = load_model(MODEL)

# training and testing
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
for i in range(10):
	history = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)
	# save trained model for future usage
	model.save("alexd3_model_01" + str(i) + ".h5")

# evaluate model
# score = model.evaluate(x_test, y_test)
# print('test loss: ', score[0])
# print('test accuracy: ', score[1])

print "--- %s seconds ---" % (time.time() - start_time)


