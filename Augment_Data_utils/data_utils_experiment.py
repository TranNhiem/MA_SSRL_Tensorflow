'''
Implementation tf.data.TFRecordDataset loading and processing data 
Reference this processing pipeline 
https://github.com/AdityaKane2001/regnety/blob/main/regnety/dataset/imagenet.py 

'''

from typing import Union, Callable, Tuple, List, Type
from datetime import datetime
import math
import tensorflow as tf
import tensorflow_addons as tfa
import os
import tensorflow_probability as tfp


AUTO = tf.data.AUTOTUNE
args.

class dataset_pipeline():

    """
    Class for all  dataset_pipeline data-related functions, including TFRecord
    parsing along with Self-Supervised augmentation transforms. TFRecords must follow the format
    given in _TFRECS_FORMAT. 
    
    
    Multi - augmentations Pipeline:
    - SimCLR + BYOL Augmentation Pipeline
    - Supervised RandAug - AutoAugmentation -- FastAuto Augmentation
    - 

    If `augment_fn` argument is not set to the string "default", it should be set to
    a callable object. That callable must take exactly two arguments: `image` and `target`
    and must return two values corresponding to the same.

    If `augment_fn` argument is 'val', then the images will be center cropped to 224x224.

    Args:
       Image_path: path for reading image
       no_aug: If True, overrides cfg and returns images as they are. Requires cfg object 
            to determine batch_size, image_size, etc.
    """

    def __init__(self, args, no_aug=False):
    
        self.tfrecs_filepath = args.tfrecs_filepath
        self.batch_size = args.batch_size
        self.image_size = args.image_size

    # this function Corresponding reading TFRecord dataset example
    def decode_example(self, example_: tf.Tensor) -> dict:
        """Decodes an example to its individual attributes.
        Args:
            example: A TFRecord dataset example.
        Returns:
            Dict containing attributes from a single example. Follows
            the same names as _TFRECS_FORMAT.
        """

        example = tf.io.parse_example(example_, _TFRECS_FORMAT)
        image = tf.reshape(
            tf.io.decode_jpeg(example["image"]), (self.image_size, self.image_size, 3)
        )
        height = example["height"]
        width = example["width"]
        filename = example["filename"]
        label = example["label"]
        synset = example["synset"]
        return {
            "image": image,
            "height": height,
            "width": width,
            "filename": filename,
            "label": label,
            "synset": synset,
        }

    
    # Taking the file_paths reading parsing image return Image_Batch

    def _read_tfrecs(self) -> Type[tf.data.Dataset]:

        """ Function for reading and loading TFRecords into a tf.data.Dataset.
        Args: None.
        Returns:
            A tf.data.Dataset instance.
        """

        files = tf.data.Dataset.list_files(self.tfrecs_filepath)
        ds = files.interleave(
            tf.data.TFRecordDataset, num_parallel_calls=AUTO, deterministic=False
        )
        ds = ds.map(self.decode_example, num_parallel_calls=AUTO)

        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.prefetch(AUTO)

        return ds

