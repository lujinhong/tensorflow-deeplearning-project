# -*- coding: utf-8 -*- 

"""
    AUTHOR: lujinhong
CREATED ON: 2021年09月16日 17:02
   PROJECT: tensorflow-deeplearning-project
   DESCRIPTION: TODO
"""


import os,sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import optimizers,models,layers,regularizers
import pandas as pd
sys.path.append("../utils")

from image.utils.mini_imagenet_preprocess import MiniImagenetPreprocess

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

batch_size = 32
input_shape = [224, 224, 3]
image_dir = '/home/ljhn1829/jupyter_data/mini-imagenet/*.jpg' if os.path.isdir('/home/ljhn1829') \
    else '/Users/lujinhong/Desktop/Dataset/mini-imagenet2/*.jpg'
model_path = 'models/vgg16_miniimagenet_subclass'


class VGG16(models.Model):
    def __init__(self, num_classes):

        super(VGG16, self).__init__()
        weight_decay = 0.000

        model = keras.models.Sequential()
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                                input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

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
        model.add(layers.Dense(num_classes, activation='softmax'))

        self.model = model

    def call(self, x):
        y = self.model(x)
        return y

    def get_model(self):
        return self.model


def print_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(10,5)) # history.history
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()


def print_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(10,5)) # history.history
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()


def train():
    mip = MiniImagenetPreprocess()
    ds_all = mip.construct_dataset(input_shape, batch_size)
    model = VGG16(mip.num_classes)
    if os.path.exists(model_path):
        model.load_weights(model_path).expect_partial()

    # 训练模型
    optimizer = optimizers.Adam(learning_rate=0.001)
    # 如果label已onehot，则loss使用categorical_crossentropy或者mse， mertircs使用categorical_accuracy
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

    model.model.summary()
    # 如果label未onehot，则loss使用sparse_categorical_crossentropy，metrics用sparse_categorical_accuracy。
    #     model.compile(loss='sparse_categorical_crossentropy',
    #              optimizer=optimizer,
    #              metrics=['sparse_categorical_accuracy', 'categorical_accuracy'])


    # 注意这里的take()和skip()的参数是经过batch后的批次大小。
    ds_train = ds_all.take(int(50000/batch_size))
    ds_valid = ds_all.skip(int(50000/batch_size))

    history = model.fit(ds_train, validation_data=ds_valid, epochs=10)
    model.summary()
    model.save(model_path)

    return model, history


if __name__ == '__main__':
    model,history = train()
    print_learning_curves(history)
