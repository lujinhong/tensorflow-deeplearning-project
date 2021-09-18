# -*- coding: utf-8 -*- 

"""
    AUTHOR: lujinhong
CREATED ON: 2021年09月17日 17:14
   PROJECT: tensorflow-deeplearning-project
   DESCRIPTION: TODO
"""


# -*- coding: utf-8 -*-

"""
    AUTHOR: lujinhong
CREATED ON: 2021年09月16日 16:52
   PROJECT: tensorflow-deeplearning-project
   DESCRIPTION: TODO
"""


import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
model_path = 'models/resnet152_fashionmnist_funtionalapi'

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


def get_resnet_model():
    inputs = keras.Input(shape=input_shape)
    output = keras.layers.Conv2D(64, 7, strides=2, input_shape=input_shape, padding='same', use_bias=False)(inputs)
    output = keras.layers.BatchNormalization()(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(output)

    pre_filter = 64
    for filters in [256]*3 + [512]*8 + [1024]*36 + [2048]*3:
        strides = 1 if filters == pre_filter else 2
        output = ResidualUnit(filters, strides)(output)
        pre_filter = filters

    output = keras.layers.GlobalAvgPool2D()(output)
    output = keras.layers.Flatten()(output)
    output = keras.layers.Dense(num_classes, activation='softmax')(output)
    return keras.Model(inputs,output)


def print_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(10,5)) # history.history
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()


def train():
    model = get_resnet_model()
    if os.path.exists(model_path):
        model.load_weights(model_path).expect_partial()
        #     如果只是加载模型用于预测，那可以直接load()。但如果还要再次保存模型，则需要使用load_weights()，get_restnet_model()用于获得网络结构，否则再次load()的时候会出错。
    #     model = keras.models.load_model(model_path)

    # 训练模型
    optimizer = optimizers.Adam(learning_rate=0.001)
    # 如果label已onehot，则loss使用categorical_crossentropy或者mse， mertircs使用categorical_accuracy
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy'])

    model.summary()
    # 如果label未onehot，则loss使用sparse_categorical_crossentropy，metrics用sparse_categorical_accuracy。
    #     model.compile(loss='sparse_categorical_crossentropy',
    #              optimizer=optimizer,
    #              metrics=['sparse_categorical_accuracy', 'categorical_accuracy'])

    history = model.fit(x_train,y_train,epochs=2,validation_data=(x_valid, y_valid))
    model.save(model_path)
    return model, history


if __name__ == '__main__':
    model,history = train()
    print_learning_curves(history)

