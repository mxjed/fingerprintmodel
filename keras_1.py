import os
import numpy as np 
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers

img_width, img_height = 224, 224
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
batch = 15

train_datagen = ImageDataGenerator(
		rescale = 1./255,
		width_shift_range=0.2,
		height_shift_range=0.2,
		rotation_range=30,
		zoom_range=0.2,
		shear_range=0.2)

validation_datagen = ImageDataGenerator(rescale = 1./255,)

		
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch,
		color_mode= 'grayscale',
        class_mode='categorical',
		)
		

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch,
		color_mode= 'grayscale',
        class_mode='categorical')

		
model = Sequential()
model.add(Convolution2D(16, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

          


nb_epoch = 200
nb_train_samples = 1734
nb_validation_samples = 481

model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)
		
model.save('model.h5')