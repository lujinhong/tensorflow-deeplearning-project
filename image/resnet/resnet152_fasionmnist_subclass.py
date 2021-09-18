# -*- coding: utf-8 -*- 

"""
    AUTHOR: lujinhong
CREATED ON: 2021年09月16日 16:52
   PROJECT: tensorflow-deeplearning-project
   DESCRIPTION: TODO
"""


import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers,regularizers,optimizers,models

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
model_path = 'models/resnet152_fashionmnist_subclass'


num_classes = 10
batch_size = 32
input_shape = [28, 28, 1]


fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000].astype(float),x_train_all[5000:].astype(float)
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]
x_train = x_train.reshape(55000, 28, 28, 1)
x_valid = x_valid.reshape(5000, 28, 28, 1)


class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super(ResidualUnit, self).__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters/4, 1, strides=strides, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters/4, 3, strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 1, strides=1, padding='same', use_bias=False),
        ]
        self.skip_layers = []
        # 每个几个RU，filters会翻倍(pre_filter*x)，特征图的长度和宽度会减半(strides=2)。
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=False),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs, **kwargs):
        main_y = inputs
        for layer in self.main_layers:
            main_y = layer(main_y)
        skip_y = inputs
        for layer in self.skip_layers:
            skip_y = layer(skip_y)
        return self.activation(main_y+skip_y)


class ResNet(models.Model):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = keras.models.Sequential()
        # 前置模块
        self.model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=input_shape, padding='same', use_bias=False))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same'))

        # 残差模块
        pre_filter = 64
        for filters in [256]*3 + [512]*8 + [1024]*36 + [2048]*3:
            strides = 1 if filters == pre_filter else 2
            self.model.add(ResidualUnit(filters, strides))
            pre_filter = filters

        # 后置模块
        self.model.add(keras.layers.GlobalAvgPool2D())
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(num_classes, activation='softmax'))

    def call(self, inputs, training=None, mask=None):
        y = self.model(inputs)
        return y


def train():

    model = ResNet().model
    if os.path.exists(model_path):
        model.load_weights(model_path).expect_partial()

    # 训练模型
    optimizer = optimizers.Adam(learning_rate=0.001)
    # 如果label已onehot，则loss使用categorical_crossentropy或者mse， mertircs使用categorical_accuracy
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy'])

    model.build(input_shape=[None] + input_shape)
    model.summary()
    # 如果label未onehot，则loss使用sparse_categorical_crossentropy，metrics用sparse_categorical_accuracy。
    #     model.compile(loss='sparse_categorical_crossentropy',
    #              optimizer=optimizer,
    #              metrics=['sparse_categorical_accuracy', 'categorical_accuracy'])

    history = model.fit(x_train,y_train,epochs=10,validation_data=(x_valid, y_valid))
    model.save(model_path)

    return model,history


if __name__ == '__main__':
    model,history = train()
