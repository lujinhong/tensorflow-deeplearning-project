# -*- coding: utf-8 -*- 

"""
    AUTHOR: lujinhong
CREATED ON: 2021年09月18日 12:24
   PROJECT: tensorflow-deeplearning-project
   DESCRIPTION: TODO
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers,optimizers,models


class VGG16(models.Model):
    def __init__(self):

        super(VGG16, self).__init__()
        weight_decay = 0.000
        num_classes = 10
        input_shape = (32, 32, 3)

        model = keras.models.Sequential()
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                                input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Flatten())
        model.add(layers.Dense(4096,kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(4096,kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        #         model.add(layers.Dense(1000,kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        #         model.add(layers.BatchNormalization())
        #         model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes))

        self.model = model

    def call(self, x):
        y = self.model(x)
        return y


# y预处理
def y_preprocess(y_train, y_test):
    # 删除y的一个维度，从[b,1]变成[b,]，否则做onehot后纬度会变成[b,1,10]，而不是[b,10]
    y_train = tf.squeeze(y_train, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    # onehot
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)
    return y_train, y_test


# x预处理
def x_preprocess(x_train, x_test):
    #数据标准化
    x_train = x_train/255.0
    x_test = x_test/255.0

    x_mean = x_train.mean()
    x_std = x_train.std()
    x_train = (x_train-x_mean)/x_std
    x_test = (x_test-x_mean)/x_std
    print(x_train.max(), x_train.min(), x_train.mean(), x_train.std())
    print(x_test.max(), x_test.min(), x_test.mean(), x_test.std())
    # 改成float32加快训练速度，避免使用float64
    #     x_train = x_train.astype(np.float32)
    #     x_test = x_test.astype(np.float32)
    return x_train, x_test


def train():

    model = VGG16()

    # 加载数据
    (x_train, y_train),(x_test, y_test) = keras.datasets.cifar10.load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    y_train, y_test = y_preprocess(y_train, y_test)
    x_train, x_test = x_preprocess(x_train, x_test)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # 训练模型
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=[keras.metrics.CategoricalAccuracy()])

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)

    # 最后一行代码也可以使用dataset的方式代替
    # ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # #必须batch()，否则少了以为，shapeError。
    # ds_train = ds_train.shuffle(50000).batch(32).repeat(1)
    # ds_test = ds_test.shuffle(50000).batch(32).repeat(1)
    # model.fit(ds_train, validation_data=ds_test, epochs=20)


if __name__ == '__main__':
    train()