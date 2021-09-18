# -*- coding: utf-8 -*- 

"""
    AUTHOR: lujinhong
CREATED ON: 2021年09月18日 14:12
   PROJECT: tensorflow-deeplearning-project
   DESCRIPTION: TODO
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, losses

(x_train, y_train),(x_test, y_test) = datasets.mnist.load_data()
print('datasets:', x_train.shape, x_train.shape, x_train.min(), x_train.max())

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train = x_train/255.0
x_test = x_test/255.0

y_train = np.array(pd.get_dummies(y_train))
y_test = np.array(pd.get_dummies(y_test))


# CNN不加L2
model = keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), input_shape=(28,28,1), padding='same', activation='sigmoid'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='sigmoid'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=120, kernel_size=(5,5), padding='valid', activation='sigmoid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(84, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.build(input_shape=(None, 28, 28, 1))
model.summary()

def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return x_train[idx], y_train[idx]

def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                          for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics,
          end=end)

n_epochs = 5
batch_size = 32
n_steps = len(x_train) // batch_size
optimizer = keras.optimizers.Nadam(learning_rate=0.01)
loss_fn = keras.losses.mean_squared_error
mean_loss = keras.metrics.Mean()

# 补充一个关于accuracy的说明：
# * 如果模型输出是一个数值，而标签也是一个数值，此时可以直接使用metrics=['accuracy']或者metrics= [keras.metrics.Accuracy()]
# * 如果模型输出是一个多分类的概率列表，标签是一个onehot后的列表，则此时使用category_accuracy或者keras.metrics.CategoricalAccuracy()
# * 如果模型输出是一个多分类的概率列表，标签是一个数值，则此时使用sparse_category_accuracy或者keras.metrics.SparseCategoricalAccuracy()

metrics = [keras.metrics.MeanAbsoluteError(), keras.metrics.CategoricalAccuracy(), keras.metrics.Precision()]

for epoch in range(1, n_epochs + 1):
    print("Epoch {}/{}".format(epoch, n_epochs))
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(x_train, y_train)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        #如果model各层中有加入正则化：
        for variable in model.variables:
            if variable.constraint is not None:
                variable.assign(variable.constraint(variable))
        mean_loss(loss)
        for metric in metrics:
            metric(y_batch, y_pred)
        print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)
    print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
    for metric in [mean_loss] + metrics:
        metric.reset_states()


# 测试误差
print('Test metrics: ')
y_test_pred = model(x_test)
for metric in metrics:
    print(metric.name, metric(y_test, y_test_pred).numpy())
