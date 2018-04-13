from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# AlexNet model structure
def build_model(input_shape):
	model = Sequential()
	model.add(Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
	model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
	model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(15, activation='softmax'))

	return model

def build_model_d3(input_shape):
	model = Sequential()
	model.add(Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(76, (3, 3), strides=(1, 1), padding='same', activation='relu'))
	model.add(Conv2D(76, (3, 3), strides=(1, 1), padding='same', activation='relu'))
	model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(13, activation='softmax'))

	return model