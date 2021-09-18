# -*- coding: utf-8 -*- 

"""
    AUTHOR: lujinhong
CREATED ON: 2021年09月18日 09:45
   PROJECT: tensorflow-deeplearning-project
   DESCRIPTION: mini-imagenet的数据预处理，返回一个预处理后的dataset。
"""

import tensorflow as tf
import os


class MiniImagenetPreprocess():
    def __init__(self):
        self.num_classes = 100
        self.image_dir = '/home/ljhn1829/jupyter_data/mini-imagenet/*.jpg' if os.path.isdir('/home/ljhn1829') \
            else '/Users/lujinhong/Desktop/Dataset/mini-imagenet2/*.jpg'

    # 提取文件名的前8位数字作为类别，然后映射到0-100个id作为分类的类别
    def construct_label_tabel(self,image_dir):
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

    # 图像的预处理：裁剪、翻转等
    def image_preproccess(self,image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)
        #     print(image.shape, type(image))
        image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
        random_s = tf.random.uniform([1], minval=int(self.input_shape[0]*1.2), maxval=int(self.input_shape[0]*1.5), dtype=tf.int32)[0]
        resized_height, resized_width = tf.cond(image_height<image_width,
                                                lambda: (random_s, tf.cast(tf.multiply(tf.cast(image_width, tf.float64),tf.divide(random_s,image_height)), tf.int32)),
                                                lambda: (tf.cast(tf.multiply(tf.cast(image_height, tf.float64),tf.divide(random_s,image_width)), tf.int32), random_s))
        image_float = tf.image.convert_image_dtype(image, tf.float32)
        image_resized = tf.image.resize(image_float, [resized_height, resized_width])

        image_flipped = tf.image.random_flip_left_right(image_resized)
        image_cropped = tf.image.random_crop(image_flipped, self.input_shape)
        image_distorted = tf.image.random_brightness(image_cropped, max_delta=0.5)
        image_distorted = tf.image.random_contrast(image_distorted, lower=0.2, upper=1.8)
        image_distorted = tf.image.per_image_standardization(image_distorted)
        image_distorted = tf.image.transpose(image_distorted)

        label_name = image_file.numpy().decode().split('/')[-1][1:9]
        label_id = self.table.lookup(tf.constant([label_name]))
        one_hot_label = tf.one_hot(int(label_id), depth=self.num_classes, dtype=tf.int64)

        #     return image_distorted,label_id.numpy()[0]
        return image_distorted, one_hot_label

    # 返回一个处理后的dataset
    def construct_dataset(self, input_shape, batch_size):
        self.table = self.construct_label_tabel(self.image_dir)
        self.input_shape = input_shape
        image_list = tf.data.Dataset.list_files(self.image_dir)
        dataset_train = image_list.map(lambda x: tf.py_function(self.image_preproccess, [x], [tf.float32,tf.int64]), num_parallel_calls=4)
        dataset_train = dataset_train.repeat(1)
        dataset_train.shuffle(10000)
        dataset_train = dataset_train.batch(batch_size)
        dataset_train = dataset_train.prefetch(batch_size)
        return dataset_train