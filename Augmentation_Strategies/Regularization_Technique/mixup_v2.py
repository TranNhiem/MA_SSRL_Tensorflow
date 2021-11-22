'''
This Mixup implementation simple strategy 
Refence : https://github.com/AdityaKane2001/regnety/blob/main/regnety/dataset/imagenet.py
'''

from typing import Union, Callable, Tuple, List, Type
import tensorflow as tf
import tensorflow_addons as tfa
import os
import tensorflow_probability as tfp

tfd = tfp.distribution


def _mixup( image, label) -> Tuple:
    """
    Function to apply mixup augmentation. To be applied after
    one hot encoding and before batching.
    Args:
        entry1: Entry from first dataset. Should be one hot encoded and batched.
        entry2: Entry from second dataset. Must be one hot encoded and batched.
    Returns:
        Tuple with same structure as the entries.
    """
    image1, label1 = image, label
    image2, label2 = tf.reverse(image, axis=[0]), tf.reverse(label, axis=[0])

    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    alpha = [0.2]
    dist = tfd.Beta(alpha, alpha)
    l = dist.sample(1)[0][0]

    img = l * image1 + (1 - l) * image2
    lab = l * label1 + (1 - l) * label2

    img = tf.cast(img, tf.uint8)

    return img, lab
