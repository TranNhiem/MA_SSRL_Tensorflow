from config.non_contrast_config_v1 import read_cfg_base
from config.absl_mock import Mock_Flag
from Augment_Data_utils.imagenet_dataloader_under_development import Imagenet_dataset
import numpy as np
from absl import logging
import tensorflow as tf
import os
from imutils import paths
from absl import flags
import random
from sklearn.preprocessing import OneHotEncoder
import time
import glob
import os

import matplotlib.pyplot as plt

from tensorflow import distribute as tf_dis
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:

    try:
        tf.config.experimental.set_visible_devices(gpus[7], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)


# dummy assignment, so let it in one line
flag = read_cfg_base()
FLAGS = flag.FLAGS


class Runner(object):
    def __init__(self, FLAGS, wanda_cfg=None):

        def infer_ds_info(n_tra_sample, n_evl_sample, train_global_batch, val_global_batch):
            self.train_steps = self.eval_steps or int(
                n_tra_sample * self.train_epochs // train_global_batch)*2
            self.epoch_steps = round(n_tra_sample / train_global_batch)

            # logging the ds info
            logging.info(f"# Subset_training class {self.num_classes}")
            logging.info(f"# train examples: {n_tra_sample}")
            logging.info(f"# eval examples: {n_evl_sample}")
            logging.info(f"# train_steps: {self.train_steps}")
            logging.info(f"# eval steps: {self.eval_steps}")

        # we use 'self' to access config in this class.. (not good, but i take it)
        self.__dict__ = FLAGS.__dict__

        # 1. Prepare imagenet dataset
        strategy = tf_dis.MirroredStrategy()
        self.strategy = strategy
        train_global_batch = self.train_batch_size * strategy.num_replicas_in_sync
        val_global_batch = self.val_batch_size * strategy.num_replicas_in_sync
        ds_args = {'img_size': self.image_size, 'train_path': self.train_path, 'val_path': self.val_path,
                   'train_label': self.train_label, 'val_label': self.val_label, 'subset_class_num': self.num_classes,
                   'train_batch': train_global_batch, 'val_batch': val_global_batch, 'strategy': strategy, 'seed': self.SEED}
        # Dataloader V2 already be proposed as formal data_loader
        train_dataset = Imagenet_dataset(**ds_args)

        n_tra_sample, n_evl_sample = train_dataset.get_data_size()
        infer_ds_info(n_tra_sample, n_evl_sample,
                      train_global_batch, val_global_batch)

        # record var into self
        self.strategy = strategy
        self.train_global_batch, self.val_global_batch = train_global_batch, val_global_batch
        self.n_tra_sample = n_tra_sample
        self.train_dataset = train_dataset


 # perform data_augmentation by calling the dataloader methods
da_crp_key = 'rnd_crp'  # incpt_crp
object_data = Runner(FLAGS, )
train_dataset = object_data.train_dataset
# train_ds = train_dataset.RandAug_strategy(crop_type=da_crp_key,
#                                           num_transform=2, magnitude=7)
train_ds = train_dataset.AutoAug_strategy(crop_type=da_crp_key)

#  already complete, have fun ~
# train_ds = self.train_dataset.FastAug_strategy(
#     crop_type=da_crp_key, policy_type="imagenet")

# #   performing Linear-protocol
val_ds = train_dataset.supervised_validation()

ds = []
for _, (ds_one, ds_two) in enumerate(train_ds):
    ds = ds_one

    break

# image_mask, lable = ds
# image = image_mask[0]
# mask = image_mask[1]

# plt.figure(figsize=(10, 5))
# for n in range(10):
#     ax = plt.subplot(2, 10, n + 1)
#     plt.imshow(image[n])  # .numpy().astype("int")
#     ax = plt.subplot(2, 10, n + 11)
#     plt.imshow(tf.squeeze(mask[n])/255)  # .numpy().astype("int")
#     plt.axis("off")
# plt.show()
# print(image[0])

image, lable = ds

plt.figure(figsize=(10, 5))
for n in range(3):
    ax = plt.subplot(2, 3, n + 1)
    # plt.imshow(image[n])  # .numpy().astype("int")
    # ax = plt.subplot(2, 10, n + 11)
    #plt.imshow(tf.squeeze(image[n])/255)  # .numpy().astype("int")
    plt.axis("off")
plt.show()
print(image[1])
# plt.imshow(image[1])
