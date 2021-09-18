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
model_path = 'models/googlenet_miniimagenet_subclass'

class Inception(layers.Layer):
    def __init__(self, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.branch1 = layers.Conv2D(ch1x1, kernel_size=1, activation="relu")

        self.branch2 = models.Sequential([
            layers.Conv2D(ch3x3red, kernel_size=1, activation="relu"),
            layers.Conv2D(ch3x3, kernel_size=3, padding="SAME", activation="relu")])      # output_size= input_size

        self.branch3 = models.Sequential([
            layers.Conv2D(ch5x5red, kernel_size=1, activation="relu"),
            layers.Conv2D(ch5x5, kernel_size=5, padding="SAME", activation="relu")])      # output_size= input_size

        self.branch4 = models.Sequential([
            layers.MaxPool2D(pool_size=3, strides=1, padding="SAME"),  # caution: default strides==pool_size
            layers.Conv2D(pool_proj, kernel_size=1, activation="relu")])                  # output_size= input_size

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)
        outputs = layers.concatenate([branch1, branch2, branch3, branch4])
        return outputs


class InceptionAux(layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super(InceptionAux, self).__init__(**kwargs)
        self.averagePool = layers.AvgPool2D(pool_size=5, strides=3)
        self.conv = layers.Conv2D(128, kernel_size=1, activation="relu")

        self.fc1 = layers.Dense(1024, activation="relu")
        self.fc2 = layers.Dense(num_classes)
        self.softmax = layers.Softmax()

    def call(self, inputs, **kwargs):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(inputs)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=0.5)(x)
        # N x 2048
        x = self.fc1(x)
        x = layers.Dropout(rate=0.5)(x)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        x = self.softmax(x)

        return x

class GoogLeNet(models.Model):
    def __init__(self, num_classes):
        aux_logits=False
        super(GoogLeNet, self).__init__()
        model = keras.models.Sequential()
        # (None, 224, 224, 3)
        model.add(layers.Conv2D(64, kernel_size=7, strides=2, padding="SAME", activation="relu", name="conv2d_1", input_shape=input_shape))
        # (None, 112, 112, 64)
        model.add(layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_1"))
        # (None, 56, 56, 64)
        model.add(layers.Conv2D(64, kernel_size=1, activation="relu", name="conv2d_2"))
        # (None, 56, 56, 64)
        model.add(layers.Conv2D(192, kernel_size=3, padding="SAME", activation="relu", name="conv2d_3"))
        # (None, 56, 56, 192)
        model.add(layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_2"))

        # (None, 28, 28, 192)
        model.add(Inception(64, 96, 128, 16, 32, 32, name="inception_3a"))
        # (None, 28, 28, 256)
        model.add(Inception(128, 128, 192, 32, 96, 64, name="inception_3b"))

        # (None, 28, 28, 480)
        model.add(layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_3"))
        # (None, 14, 14, 480)
        model.add(Inception(192, 96, 208, 16, 48, 64, name="inception_4a"))
        if aux_logits:
            aux1 = model.add(InceptionAux(num_classes, name="aux_1"))

            # (None, 14, 14, 512)
        model.add(Inception(160, 112, 224, 24, 64, 64, name="inception_4b"))
        # (None, 14, 14, 512)
        model.add(Inception(128, 128, 256, 24, 64, 64, name="inception_4c"))
        # (None, 14, 14, 512)
        model.add(Inception(112, 144, 288, 32, 64, 64, name="inception_4d"))
        if aux_logits:
            aux2 = model.add(InceptionAux(num_classes, name="aux_2"))

        # (None, 14, 14, 528)
        model.add(Inception(256, 160, 320, 32, 128, 128, name="inception_4e"))
        # (None, 14, 14, 532)
        model.add(layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_4"))

        # (None, 7, 7, 832)
        model.add(Inception(256, 160, 320, 32, 128, 128, name="inception_5a"))
        # (None, 7, 7, 832)
        model.add(Inception(384, 192, 384, 48, 128, 128, name="inception_5b"))
        # (None, 7, 7, 1024)
        model.add(layers.AvgPool2D(pool_size=7, strides=1, name="avgpool_1"))

        # (None, 1, 1, 1024)
        model.add(layers.Flatten(name="output_flatten"))
        # (None, 1024)
        model.add(layers.Dropout(rate=0.4, name="output_dropout"))
        model.add(layers.Dense(num_classes, name="output_dense", activation='softmax'))
        # (None, num_classes)
        #         model.add(layers.Softmax(name="aux_3"))

        # if aux_logits:
        #     model = models.Model(inputs=input_image, outputs=[aux1, aux2, aux3])
        # else:
        #     model = models.Model(inputs=input_image, outputs=aux3)
        self.model = model

    def call(self, x):
        y = self.model(x)
        return y


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
    model = GoogLeNet(mip.num_classes)
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

    history = model.fit(ds_train, validation_data=ds_valid, epochs=2)
    model.summary()
    model.save(model_path)

    return model, history


if __name__ == '__main__':
    model,history = train()
    print_learning_curves(history)
