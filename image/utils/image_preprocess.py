# -*- coding: utf-8 -*- 

"""
    AUTHOR: lujinhong
CREATED ON: 2021年09月09日 11:21
   PROJECT: tensorflow-deeplearning-project
   DESCRIPTION: TODO
"""

import os
import random
import tensorflow as tf

batch_size = 32

#Imagenet图片都保存在/data目录下，里面有1000个子目录，获取这些子目录的名字
classes = os.listdir('data/')

#构建一个字典，Key是目录名，value是类名0-999
labels_dict = {}
for i in range(len(classes)):
    labels_dict[classes[i]]=i

#构建一个列表，里面的每个元素是图片文件名+类名
images_labels_list = []
for i in range(len(classes)):
    path = 'data/'+classes[i]+'/'
    images_files = os.listdir(path)
    label = str(labels_dict[classes[i]])
    for image_file in images_files:
        images_labels_list.append(path+image_file+','+label+'\n')

#把列表进行随机排序，然后取其中80%的数据作为训练集，10%作为验证集，10%作为测试集
random.shuffle(images_labels_list)
num = len(images_labels_list)
with open('imagenet_train.csv', 'w') as file:
    file.writelines(images_labels_list[:int(num*0.8)])
with open('imagenet_valid.csv', 'w') as file:
    file.writelines(images_labels_list[int(num*0.8):int(num*0.9)])
with open('imagenet_test.csv', 'w') as file:
    file.writelines(images_labels_list[int(num*0.9):])


#定义对Dataset每条数据进行处理的map函数
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_height = tf.shape(image_decoded)[0]
    image_width = tf.shape(image_decoded)[1]
    #按照RESNET论文的训练图像的处理方式，对图片的短边随机缩放到256-481之间的数值，然后在随机
    #剪切224×224大小的图片。
    random_s = tf.random_uniform([1], minval=256, maxval=481, dtype=tf.int32)[0]
    resized_height, resized_width = tf.cond(image_height<image_width,
                                            lambda: (random_s, tf.cast(tf.multiply(tf.cast(image_width, tf.float64),tf.divide(random_s,image_height)), tf.int32)),
                                            lambda: (tf.cast(tf.multiply(tf.cast(image_height, tf.float64),tf.divide(random_s,image_width)), tf.int32), random_s))
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize_images(image_float, [resized_height, resized_width])
    image_flipped = tf.image.random_flip_left_right(image_resized)
    image_cropped = tf.random_crop(image_flipped, [imageCropHeight, imageCropWidth, imageDepth])
    image_distorted = tf.image.random_brightness(image_cropped, max_delta=63)
    image_distorted = tf.image.random_contrast(image_distorted, lower=0.2, upper=1.8)
    image_distorted = tf.image.per_image_standardization(image_distorted)
    image_distorted = tf.transpose(image_distorted, perm=[2, 0, 1])

    return image_distorted, label

#构建Dataset
with tf.device('/cpu:0'):
    filename_train = ["imagenet_train.csv"]
    filename_valid = ["imagenet_valid.csv"]
    #filename_test = ["imagenet_test.csv"]
    record_defaults = [tf.string, tf.int32]
    dataset_train = tf.contrib.data.CsvDataset(filename_train, record_defaults)
    dataset_valid = tf.contrib.data.CsvDataset(filename_valid, record_defaults)
    #dataset_test = tf.contrib.data.CsvDataset(filename_test, record_defaults)
    dataset_train = dataset_train.map(_parse_function, num_parallel_calls=4)
    dataset_valid = dataset_valid.map(_parse_test_function, num_parallel_calls=2)
    #dataset_test = dataset_test.map(_parse_function, num_parallel_calls=2)

    dataset_train = dataset_train.repeat(10)

    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(batch_size)
    dataset_valid = dataset_valid.batch(batch_size)
    #dataset_test = dataset_test.batch(batch_size)

    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    next_images, next_labels = iterator.get_next()
    train_init_op = iterator.make_initializer(dataset_train)
    valid_init_op = iterator.make_initializer(dataset_valid)
    #test_init_op = iterator.make_initializer(dataset_test)


