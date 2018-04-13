import numpy as np
from numpy import genfromtxt
from PIL import Image
import keras
from keras.optimizers import Adam
from keras.models import load_model
import vgg11
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
# x_train = np.zeros((12762, 32*32*3)).astype(int)
# y_train = np.zeros((12762)).astype(int)

# print "loading data..."
# path = "color/"
# i = 0
# for filename in os.listdir(path):
# 	img = Image.open(path+filename)
# 	x_train[i] = np.array(img).reshape((1,32*32*3))
# 	token = filename.split("_")
# 	y_train[i] = int(token[0])
# 	i += 1
# print i

# print "saving data..."
# np.savetxt('x_sub_color.csv', x_train, fmt='%d', delimiter=',')
# np.savetxt('y_sub_color.csv', y_train, fmt='%d', delimiter=',')
# ========== process image data into csv file ==========
# print "--- %s seconds ---" % (time.time() - start_time)

print "loading data..."
x_train = genfromtxt("x_sub_color.csv", delimiter=',')
y_train = genfromtxt("y_sub_color.csv", delimiter=',')

x_train = scale(x_train)
x_train = x_train.reshape((12762, 32, 32, 3))
y_train = keras.utils.to_categorical(y_train, 13)

print "training..."
# build vgg11 model
# input_shape = (32, 32, 3)
# model = vgg11.build_model(input_shape)

MODEL = "sub_vgg_colo_009.h5"
# print MODEL
model = load_model(MODEL)

# training and testing
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
for i in range(10):
	history = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)
	# save trained model for future usage
	model.save("sub_vgg_colo_01" + str(i) + ".h5")

# evaluate model
# score = model.evaluate(x_test, y_test)
# print('test loss: ', score[0])
# print('test accuracy: ', score[1])

print "--- %s seconds ---" % (time.time() - start_time)


