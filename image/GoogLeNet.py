# -*- coding: utf-8 -*- 

"""
    AUTHOR: lujinhong
CREATED ON: 2021年09月14日 09:33
   PROJECT: tensorflow-deeplearning-project
   DESCRIPTION: 构建googlenet并为mini-imagenet分类。
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers,optimizers,models


# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from tensorflow.python.keras.models import Sequential

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

num_classes = 100
batch_size = 32
input_shape = [224, 224, 3]
# image_dir = '/home/ljhn1829/jupyter_data/mini-imagenet/*.jpg'
image_dir = '/Users/lujinhong/Desktop/Dataset/mini-imagenet2/*.jpg'


def construct_label_tabel(image_dir):
    image_list = tf.data.Dataset.list_files(image_dir)
    label_set = set()
    for image in image_list:
        label_set.add(str(image.numpy()).split('/')[-1][1:9])

    label_list = list(label_set)
    indices = tf.range(len(label_list), dtype=tf.int64)
    table_init = tf.lookup.KeyValueTensorInitializer(label_list, indices)
    num_oov_buckets = 2
    table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)
    return table


def image_preproccess(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
    random_s = tf.random.uniform([1], minval=int(input_shape[0]*1.2), maxval=int(input_shape[0]*1.5), dtype=tf.int32)[0]
    resized_height, resized_width = tf.cond(image_height<image_width,
                                            lambda: (random_s, tf.cast(tf.multiply(tf.cast(image_width, tf.float64),tf.divide(random_s,image_height)), tf.int32)),
                                            lambda: (tf.cast(tf.multiply(tf.cast(image_height, tf.float64),tf.divide(random_s,image_width)), tf.int32), random_s))
    image_float = tf.image.convert_image_dtype(image, tf.float32)
    image_resized = tf.image.resize(image_float, [resized_height, resized_width])

    image_flipped = tf.image.random_flip_left_right(image_resized)
    image_cropped = tf.image.random_crop(image_flipped, input_shape)
    image_distorted = tf.image.random_brightness(image_cropped, max_delta=0.5)
    image_distorted = tf.image.random_contrast(image_distorted, lower=0.2, upper=1.8)
    image_distorted = tf.image.per_image_standardization(image_distorted)
    image_distorted = tf.image.transpose(image_distorted)

    label_name = image_file.numpy().decode().split('/')[-1][1:9]
    label_id = table.lookup(tf.constant([label_name]))
    one_hot_label = tf.one_hot(int(label_id), depth=num_classes, dtype=tf.int64)

    #     return image_distorted,label_id.numpy()[0]
    return image_distorted, one_hot_label


# 构建dataset
def construct_dataset():
    #     construct_label_tabel(image_dir)
    image_list = tf.data.Dataset.list_files(image_dir)
    dataset_train = image_list.map(lambda x: tf.py_function(image_preproccess, [x], [tf.float32,tf.int64]), num_parallel_calls=4)
    dataset_train = dataset_train.repeat(1)
    dataset_train.shuffle(10000)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(batch_size)
    return dataset_train


# def GoogLeNet(im_height=input_shape[0], im_width=input_shape[1], class_num=num_classes, aux_logits=False):
#     # tensorflow中的tensor通道排序是NHWC
#     input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
#     # (None, 224, 224, 3)
#     x = layers.Conv2D(64, kernel_size=7, strides=2, padding="SAME", activation="relu", name="conv2d_1")(input_image)
#     # (None, 112, 112, 64)
#     x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_1")(x)
#     # (None, 56, 56, 64)
#     x = layers.Conv2D(64, kernel_size=1, activation="relu", name="conv2d_2")(x)
#     # (None, 56, 56, 64)
#     x = layers.Conv2D(192, kernel_size=3, padding="SAME", activation="relu", name="conv2d_3")(x)
#     # (None, 56, 56, 192)
#     x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_2")(x)
#
#     # (None, 28, 28, 192)
#     x = Inception(64, 96, 128, 16, 32, 32, name="inception_3a")(x)
#     # (None, 28, 28, 256)
#     x = Inception(128, 128, 192, 32, 96, 64, name="inception_3b")(x)
#
#     # (None, 28, 28, 480)
#     x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_3")(x)
#     # (None, 14, 14, 480)
#     x = Inception(192, 96, 208, 16, 48, 64, name="inception_4a")(x)
#     if aux_logits:
#         aux1 = InceptionAux(class_num, name="aux_1")(x)
#
#     # (None, 14, 14, 512)
#     x = Inception(160, 112, 224, 24, 64, 64, name="inception_4b")(x)
#     # (None, 14, 14, 512)
#     x = Inception(128, 128, 256, 24, 64, 64, name="inception_4c")(x)
#     # (None, 14, 14, 512)
#     x = Inception(112, 144, 288, 32, 64, 64, name="inception_4d")(x)
#     if aux_logits:
#         aux2 = InceptionAux(class_num, name="aux_2")(x)
#
#     # (None, 14, 14, 528)
#     x = Inception(256, 160, 320, 32, 128, 128, name="inception_4e")(x)
#     # (None, 14, 14, 532)
#     x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_4")(x)
#
#     # (None, 7, 7, 832)
#     x = Inception(256, 160, 320, 32, 128, 128, name="inception_5a")(x)
#     # (None, 7, 7, 832)
#     x = Inception(384, 192, 384, 48, 128, 128, name="inception_5b")(x)
#     # (None, 7, 7, 1024)
#     x = layers.AvgPool2D(pool_size=7, strides=1, name="avgpool_1")(x)
#
#     # (None, 1, 1, 1024)
#     x = layers.Flatten(name="output_flatten")(x)
#     # (None, 1024)
#     x = layers.Dropout(rate=0.4, name="output_dropout")(x)
#     x = layers.Dense(class_num, name="output_dense")(x)
#     # (None, class_num)
#     aux3 = layers.Softmax(name="aux_3")(x)
#
#     if aux_logits:
#         model = models.Model(inputs=input_image, outputs=[aux1, aux2, aux3])
#     else:
#         model = models.Model(inputs=input_image, outputs=aux3)
#     return model


class Inception(layers.Layer):
    def __init__(self, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.branch1 = layers.Conv2D(ch1x1, kernel_size=1, activation="relu")

        self.branch2 = Sequential([
            layers.Conv2D(ch3x3red, kernel_size=1, activation="relu"),
            layers.Conv2D(ch3x3, kernel_size=3, padding="SAME", activation="relu")])      # output_size= input_size

        self.branch3 = Sequential([
            layers.Conv2D(ch5x5red, kernel_size=1, activation="relu"),
            layers.Conv2D(ch5x5, kernel_size=5, padding="SAME", activation="relu")])      # output_size= input_size

        self.branch4 = Sequential([
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






# 即可以使用函数，也可以使用类的方式
class GoogLeNet(models.Model):
    def __init__(self):
        aux_logits=False
        super(GoogLeNet, self).__init__()
        model = keras.models.Sequential()
        # tensorflow中的tensor通道排序是NHWC
        # model.add(layers.Input(shape=input_shape, dtype="float32"))
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
        model.add(layers.Dense(num_classes, name="output_dense"))
        # (None, num_classes)
        model.add(layers.Softmax(name="aux_3"))

        # if aux_logits:
        #     model = models.Model(inputs=input_image, outputs=[aux1, aux2, aux3])
        # else:
        #     model = models.Model(inputs=input_image, outputs=aux3)
        self.model = model

    def call(self, x):
        y = self.model(x)
        return y


# 训练模型
if __name__ == '__main__':
    print([batch_size] + input_shape)
    table = construct_label_tabel(image_dir)

    label_id = table.lookup(tf.constant(['03476684']))
    one_hot_label = tf.one_hot(int(label_id), depth=num_classes)
    print(label_id.numpy()[0], one_hot_label)

    model = GoogLeNet().model

    # 训练模型
    optimizer = optimizers.Adam(learning_rate=0.001)
    # 如果label已onehot，则loss使用categorical_crossentropy或者mse， mertircs使用categorical_accuracy
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,  # 好像sgd最容易收敛
                  metrics=['categorical_accuracy','top_k_categorical_accuracy'])

    # 如果label未onehot，则loss使用sparse_categorical_crossentropy，metrics用sparse_categorical_accuracy。
    #     model.compile(loss='sparse_categorical_crossentropy',
    #              optimizer=optimizer,
    #              metrics=['sparse_categorical_accuracy', 'categorical_accuracy'])
    model.build(input_shape=[None] + input_shape)
    model.summary()

    ds_all = construct_dataset()

    # 注意这里的take()和skip()的参数是经过batch后的批次大小。
    ds_train = ds_all.take(int(50000/batch_size))
    ds_valid = ds_all.skip(int(50000/batch_size))

    history = model.fit(ds_train, validation_data=ds_valid, epochs=1)
    model.summary()