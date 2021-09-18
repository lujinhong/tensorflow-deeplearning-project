import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import random
from tensorflow import keras
from tensorflow.keras import layers,regularizers,optimizers,models

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

num_classes = 10
batch_size = 32
input_shape = [28, 28, 1]
model_path = 'models/zfnet_fashionmnist_subclass'


fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]
x_train = x_train.reshape(55000, 28, 28, 1)
x_valid = x_valid.reshape(5000, 28, 28, 1)


class ZFNet(models.Model):
    def __init__(self):

        super(ZFNet, self).__init__()
        weight_decay = 0.001
        #         num_classes = num_classes
        #         input_shape = input_shape

        model = keras.models.Sequential()
        model.add(layers.Conv2D(filters=96, kernel_size=(7, 7), strides=2, padding='same',
                                input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(layers.Conv2D(256, (5, 5), strides=1, padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(layers.Conv2D(384, (3, 3), strides=1, padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))

        model.add(layers.Conv2D(384, (3, 3), strides=1, padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))

        model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(layers.Flatten())
        model.add(layers.Dense(4096,kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(4096,kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
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


def train():
    #     model = ZFNet(input_shape)
    model = ZFNet()
    if os.path.exists(model_path):
        model.load_weights(model_path).expect_partial()
    # 训练模型
    optimizer = optimizers.Adam(learning_rate=0.001)
    # 如果label已onehot，则loss使用categorical_crossentropy或者mse， mertircs使用categorical_accuracy
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy'])

    #     model.build(input_shape=[None] + input_shape)
    model.model.summary()
    # 如果label未onehot，则loss使用sparse_categorical_crossentropy，metrics用sparse_categorical_accuracy。
    #     model.compile(loss='sparse_categorical_crossentropy',
    #              optimizer=optimizer,
    #              metrics=['sparse_categorical_accuracy', 'categorical_accuracy'])

    history = model.fit(x_train,y_train,epochs=10,validation_data=(x_valid,y_valid))
    model.save(model_path)
    return model,history


if __name__ == '__main__':
    model,history = train()
