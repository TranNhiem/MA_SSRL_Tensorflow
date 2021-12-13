import os
#from absl import flags
import tensorflow as tf
from imutils import paths
from "DA_Object" import simclr_augment_randcrop_global_views, simclr_augment_inception_style, \
    supervised_augment_eval, simclr_augment_randcrop_global_view_image_mask, simclr_augment_inception_style_image_mask
from absl import logging
import numpy as np
import random
import re

AUTO = tf.data.experimental.AUTOTUNE

#FLAGS = flags.FLAGS

from config.absl_mock import Mock_Flag
flag = Mock_Flag()
FLAGS = flag.FLAGS


class imagenet_dataset_single_machine():

    def __init__(self, img_size, train_batch, val_batch, strategy, train_path=None,train_label = None, val_path=None,val_label = None, bi_mask=False,
                 mask_path=None,subset_class_num = None):
        '''
        args: 
        img_size: Image training size
        train_batch: Distributed Batch_size for training multi-GPUs

        image_path: Directory to train data 
        val_path:   Directory to validation or testing data
        subset_class_num: subset class 

        '''

        self.IMG_SIZE = img_size
        self.BATCH_SIZE = train_batch
        self.val_batch = val_batch
        self.strategy = strategy
        self.seed = FLAGS.SEED
        self.bi_mask = []

        self.label, self.class_name = self.get_label(train_label)
        numeric_train_cls = []
        numeric_val_cls = []
        print("train_path:",train_path)
        print("val_path:",val_path)

        if train_path is None and val_path is None:
            raise ValueError(f'The train_path and val_path is None, please cheeek')
        elif val_path is None:
            dataset = list(paths.list_images(train_path))
            dataset_len =  len(dataset)
            random.Random(FLAGS.SEED_data_split).shuffle(dataset)
            self.x_val = dataset[0:int(dataset_len * 0.2)]
            self.x_train = dataset[len(self.x_val) + 1:]
            for image_path in self.x_train:
                label = re.split(r"/|\|//|\\",image_path)[-2]
                #label = image_path.split("/")[-2]
                numeric_train_cls.append(self.label[label])
            for image_path in self.x_val:
                label = re.split(r"/|\|//|\\",image_path)[-2]
                numeric_val_cls.append(self.label[label])

        else:
            self.x_train = list(paths.list_images(train_path))
            
            self.x_val = list(paths.list_images(val_path))
            random.Random(FLAGS.SEED_data_split).shuffle(self.x_train)
            random.Random(FLAGS.SEED_data_split).shuffle(self.x_val)

            for image_path in self.x_train:
                label = re.split(r"/|\|//|\\",image_path)[-2]
                numeric_train_cls.append(self.label[label])

            val_label_map = self.get_val_label(val_label)
            numeric_val_cls = []
            for image_path in self.x_val:
                label = re.split(r"/|\|//|\\", image_path)[-1]

                label = label.split("_")[-1]
                label = int(label.split(".")[0])
                numeric_val_cls.append(val_label_map[label-1])


        if subset_class_num != None:
            x_train_sub = []
            numeric_train_cls_sub = []
            for file_path,numeric_cls in zip(self.x_train,numeric_train_cls):
                if numeric_cls<subset_class_num:
                    x_train_sub.append(file_path)
                    numeric_train_cls_sub.append(numeric_cls)
            self.x_train = x_train_sub
            numeric_train_cls = numeric_train_cls_sub
            
            x_val_sub = []
            numeric_val_cls_sub = []
            for file_path,numeric_cls in zip(self.x_val,numeric_val_cls):
                if numeric_cls<subset_class_num:
                    x_val_sub.append(file_path)
                    numeric_val_cls_sub.append(numeric_cls)
            self.x_val = x_val_sub
            numeric_val_cls = numeric_val_cls_sub
        

        if bi_mask:
            for p in self.x_train:
                self.bi_mask.append(p.replace("train", mask_path).replace("JPEG", "png"))

        # Path for loading all Images
        # For training

        self.x_train_lable = tf.one_hot(numeric_train_cls, depth = len(self.class_name) if subset_class_num==None else subset_class_num)
        self.x_val_lable = tf.one_hot(numeric_val_cls, depth = len(self.class_name) if subset_class_num==None else subset_class_num)

        if bi_mask:
            self.x_train_image_mask = np.stack(
                (np.array(self.x_train), np.array(self.bi_mask)), axis=-1)
            print(self.x_train_image_mask.shape)
        
    def get_label(self, label_txt_path=None):
        class_name = []
        class_ID = []
        class_number = []
        print(label_txt_path)
        with open(label_txt_path) as file:
            for line in file.readlines():
                # n02119789 1 kit_fox
                lint_split = line.split(" ")
                class_ID.append(lint_split[0])
                class_number.append(int(lint_split[1]))
                class_name.append(lint_split[2])
            file.close()

        label = dict(zip(class_ID, class_number))
        class_name = dict(zip(class_ID, class_name))
        return label, class_name

    def get_val_label(self, label_txt_path=None):
        class_number = []
        with open(label_txt_path) as file:
            for line in file.readlines():
                class_number.append(int(line[:-1]))
                # n02119789 1 kit_fox
        return class_number

    @classmethod
    def parse_images(self, image_path):
        # Loading and reading Image
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    @classmethod
    def parse_images_lable_pair(self, image_path, lable):
        # Loading and reading Image
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        return img, lable

    @classmethod
    def parse_images_mask_lable_pair(self, image_mask_path, lable, IMG_SIZE):
        # Loading and reading Image
        # print(image_mask_path[0])
        # print(image_mask_path[1])
        image_path, mask_path = image_mask_path[0], image_mask_path[1]
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

        bi_mask = tf.io.read_file(mask_path)
        bi_mask = tf.io.decode_jpeg(bi_mask, channels=1)
        bi_mask = tf.image.resize(bi_mask, (IMG_SIZE, IMG_SIZE))
        return img, bi_mask, lable

    @classmethod
    def parse_images_label(self, image_path):
        img = tf.io.read_file(image_path)
        # img = tf.image.decode_jpeg(img, channels=3) # decode the image back to proper format
        img = tf.io.decode_jpeg(img, channels=3)
        label = tf.strings.split(image_path, os.path.sep)[4]
        # print(label)
        return img, label

    def supervised_validation(self):
        '''This for Supervised validation training'''

        val_ds = (tf.data.Dataset.from_tensor_slices((self.x_val, self.x_val_lable))
                  .shuffle(self.val_batch * 100, seed=self.seed)
                  .map(lambda x, y: (self.parse_images_lable_pair(x, y)), num_parallel_calls=AUTO)

                  .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                       num_parallel_calls=AUTO, )
                  .map(lambda x, y: (
        supervised_augment_eval(x, FLAGS.IMG_height, FLAGS.IMG_width, FLAGS.randaug_transform, FLAGS.randaug_magnitude),
        y), num_parallel_calls=AUTO)
                  .batch(self.BATCH_SIZE)
                  .prefetch(AUTO)
                  )

        val_ds = self.strategy.experimental_distribute_dataset(val_ds)

        return val_ds

    def simclr_inception_style_crop(self):
        '''
        This class property return self-supervised training data
        '''
        train_ds_one = (tf.data.Dataset.from_tensor_slices((self.x_train, self.x_train_lable))
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        # .map(self.parse_images_label,  num_parallel_calls=AUTO)
                        .map(lambda x, y: (self.parse_images_lable_pair(x, y)), num_parallel_calls=AUTO)
                        .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                             num_parallel_calls=AUTO,
                             )
                        .map(lambda x, y: (simclr_augment_inception_style(x, self.IMG_SIZE), y),
                             num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )
        # train_ds_one= self.strategy.experimental_distribute_dataset(train_ds_one)

        train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train, self.x_train_lable))
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        # .map(self.parse_images_label,  num_parallel_calls=AUTO)
                        .map(lambda x, y: (self.parse_images_lable_pair(x, y)), num_parallel_calls=AUTO)
                        .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                             num_parallel_calls=AUTO,
                             )
                        .map(lambda x, y: (simclr_augment_inception_style(x, self.IMG_SIZE), y),
                             num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )
        # train_ds_one= self.strategy.experimental_distribute_dataset(train_ds_two)

        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        train_ds = self.strategy.experimental_distribute_dataset(train_ds)
        # train_ds = train_ds.batch(self.BATCH_SIZE)
        # # 2. modify dataset with prefetch
        # train_ds = train_ds.prefetch(AUTO)

        return train_ds

    def simclr_random_global_crop(self):

        train_ds_one = (tf.data.Dataset.from_tensor_slices((self.x_train, self.x_train_lable))
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        # .map(self.parse_images_label,  num_parallel_calls=AUTO)
                        .map(lambda x, y: (self.parse_images_lable_pair(x, y)), num_parallel_calls=AUTO)
                        .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                             num_parallel_calls=AUTO,
                             )
                        .map(lambda x, y: (simclr_augment_randcrop_global_views(x, self.IMG_SIZE), y),
                             num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train, self.x_train_lable))
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        # .map(self.parse_images_label,  num_parallel_calls=AUTO)
                        .map(lambda x, y: (self.parse_images_lable_pair(x, y)), num_parallel_calls=AUTO)
                        .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                             num_parallel_calls=AUTO,
                             )
                        .map(lambda x, y: (simclr_augment_randcrop_global_views(x, self.IMG_SIZE), y),
                             num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        # adding the distribute data to GPUs
        train_ds = self.strategy.experimental_distribute_dataset(train_ds)

        return train_ds

    def simclr_inception_style_crop_image_mask(self):

        train_ds_one = (tf.data.Dataset.from_tensor_slices((self.x_train_image_mask, self.x_train_lable))
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        # .map(self.parse_images_label,  num_parallel_calls=AUTO)
                        .map(lambda x, y: (self.parse_images_mask_lable_pair(x, y, self.IMG_SIZE)),
                             num_parallel_calls=AUTO)
                        .map(lambda x, y, z: (simclr_augment_inception_style_image_mask(x, y, self.IMG_SIZE), z),
                             num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train_image_mask, self.x_train_lable))
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        # .map(self.parse_images_label,  num_parallel_calls=AUTO)
                        .map(lambda x, y: (self.parse_images_mask_lable_pair(x, y, self.IMG_SIZE)),
                             num_parallel_calls=AUTO)
                        .map(lambda x, y, z: (simclr_augment_inception_style_image_mask(x, y, self.IMG_SIZE), z),
                             num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )
        # train_ds_one= self.strategy.experimental_distribute_dataset(train_ds_two)

        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        # train_ds=train_ds.batch(self.BATCH_SIZE)
        # train_ds=train_ds.prefetch(AUTO)
        train_ds = self.strategy.experimental_distribute_dataset(train_ds)
        # train_ds = train_ds.batch(self.BATCH_SIZE)
        # # 2. modify dataset with prefetch
        # train_ds = train_ds.prefetch(AUTO)

        return train_ds

    def simclr_random_global_crop_image_mask(self):

        train_ds_one = (tf.data.Dataset.from_tensor_slices((self.x_train_image_mask, self.x_train_lable))
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        # .map(self.parse_images_label,  num_parallel_calls=AUTO)
                        .map(lambda x, y: (self.parse_images_mask_lable_pair(x, y, self.IMG_SIZE)),
                             num_parallel_calls=AUTO)
                        .map(lambda x, y, z: (simclr_augment_randcrop_global_view_image_mask(x, y, self.IMG_SIZE), z),
                             num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train_image_mask, self.x_train_lable))
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        # .map(self.parse_images_label,  num_parallel_calls=AUTO)
                        .map(lambda x, y: (self.parse_images_mask_lable_pair(x, y, self.IMG_SIZE)),
                             num_parallel_calls=AUTO)
                        .map(lambda x, y, z: (simclr_augment_randcrop_global_view_image_mask(x, y, self.IMG_SIZE), z),
                             num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )
        # train_ds_one= self.strategy.experimental_distribute_dataset(train_ds_two)

        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        train_ds = self.strategy.experimental_distribute_dataset(train_ds)

        return train_ds

    def get_data_size(self):
        return len(self.x_train) , len(self.x_val)




