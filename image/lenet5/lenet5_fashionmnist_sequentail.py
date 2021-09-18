# -*- coding: utf-8 -*- 

"""
    AUTHOR: lujinhong
CREATED ON: 2021年09月18日 14:06
   PROJECT: tensorflow-deeplearning-project
   DESCRIPTION: TODO
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras

(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape, y_train.shape)

x_train = x_train/255.0
x_test = x_test/255.0

y_train = np.array(pd.get_dummies(y_train))
y_test = np.array(pd.get_dummies(y_test))

model = keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), input_shape=(28,28,1), padding='same', activation='sigmoid'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='sigmoid'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=120, kernel_size=(5,5), padding='valid', activation='sigmoid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(84, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
model.save('mnist')

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# 预测
import cv2
img = cv2.imread('3.png', 0)
plt.imshow(img)

img = cv2.resize(img, (28,28))
img = img.reshape(1, 28, 28, 1)
img = img/255.0

my_model = tf.keras.models.load_model('mnist')
predict = my_model.predict(img)
print(predict)
print(np.argmax(predict))