__src_str__ = '''
Data Augumentation Searching Package :
  1. Auto Augmentation Policy Focus on Object Detection Task 
    # Reference Implementation on Object Detection with 
    # AutoAugment have 3 Policies (V0- V3)
    Barret, et al. Learning Data Augmentation Strategies for Object Detection.
        Arxiv: https://arxiv.org/abs/1906.11172
    
    ## Reference GitHub for implementation
    [1] https://github.com/google/automl/blob/master/efficientdet/aug/autoaugment.py
    [2] https://github.com/tensorflow/models/blob/master/official/vision/image_classification/augment.py

  2. RandAug implementation for Self-Supervised Learning
    paper: https://arxiv.org/abs/1909.13719
    RandAug with 2 Version : 
        Version 1 from Original Paper (14 Transformations )
        Version 2 Modify with Adding Multiple Transformation (22 transformation from Imgaug API)

  3. Fast Auto Augument
    paper: https://arxiv.org/abs/1905.00397

    ## Reference GitHub for implementation
    [1] https://github.com/kakaobrain/fast-autoaugment (official)
    [2] https://github.com/NoamRosenberg/fast-autoaugment
    [3] https://github.com/JunYeopLee/fast-autoaugment-efficientnet-pytorch (not recommend)
'''

# Current implementation will Deploy for Images WITHOUT BOX
import tensorflow as tf
from tensorflow.image import random_flip_left_right, random_crop

# flexible policies implementation of AutoAugment, RandAugment
from Fast_Auto_Augment import Fast_AutoAugment
# tf-models-official 
from tf_official_DA import AutoAugment, RandAugment
import imgaug.augmenters as iaa

# For 'Type Hint' function, we replace the comment of code into type package.. 
#from typing import Any, Dict, List, Optional, Text, Tuple
from functools import partial
import numpy as np


class Data_Augumentor(object):
    # static instance, declare once use everywhere!
    DAS_dict = {"auto_aug":AutoAugment, "tf_rand_aug":RandAugment, "iaa_rand_aug":iaa.RandAugment}
    # common method name to apply the data augmentation!!
    DA_METHOD = "distort"
    def __init__(self, DAS_type="auto_aug", *aug_args, **aug_kwarg):
        try:
            self.DAS_type = DAS_type
            self.aug_inst = DAS_dict[DAS_type](*aug_args, **aug_kwarg)
        except KeyError as k_err:
            raise KeyError("Given vlaue {} of DAS_type, \
                            but the value should be one of that ({})".format(
                            aug_type, DAS_dict.keys() )
                          )
        except Exception as exc:
            print(exc)

        self.pre_proc_dict = {"default":lambda image, *_ : tf.cast(image, dtype=tf.float32),
                    "rnd_crp_flp":self.rand_crop_flip,
                    "glb_loc_crp_flp":self.rand_distribe_crop_global_local_views_flip}
        self.post_proc = lambda image : tf.cast(image, dtype=tf.float32) / 255.
        self.regist_common_distort()

    @staticmethod
    def prnt_policies(DAS_type):
        if DAS_type == "auto_aug":
            print("AutoAugment Policy V0-- implementation : \n")
            print("V0--> policy = \n")
            print(DAS_dict[DAS_type].policy_v0())
            print("\n\n=================\n\n")
            print("Policy_simple = \n")
            print(DAS_dict[DAS_type].policy_simple())

        elif DAS_type == "tf_rand_aug":
            print("Tensorflow RandAugment avaliable ops : \n")
            print(DAS_dict[DAS_type].available_ops)

        # GG.. ImageAug did not offer any public method to get the avaliable_ops
        elif DAS_type == "iaa_rand_aug": 
            print("ImageAug RandAugment avaliable ops : \n")
            aug_str = '''[ Fliplr, KeepSizeByResize, Crop, Sequential, SomeOf, Identity, \
                            Autocontrast, Equalize, Invert, Affine, Posterize, Solarize, EnhanceColor, \
                            EnhanceContrast,EnhanceBrightness, EnhanceSharpness, Cutout, FilterBlur, FilterSmooth]'''
            print(aug_str)
        else:
            raise ValueError("Given vlaue {} of DAS_type, \
                            but the value should be one of that ({})".format(
                            aug_type, DAS_dict.keys() )
                          )

    # allow the augument instance have common method name to apply the data transformation
    def regist_common_distort(self):
        # the distort method already implemented in augument instance
        if Data_Augumentor.DA_METHOD in dir(self.aug_inst):
            return
        # plz regist the correct method, which apply the data transformation.. 
        if self.DAS_type == "iaa_rand_aug":
            self.aug_inst.distort = self.aug_inst.__call__


    def rand_crop_flip(self, image, crop_size):
        '''
            Args: 
                image: A tensor [ with, height, channels]
                crop_size: Apply Random Crop_Flip Image before Apply AutoAugment
                AutoAugment: a function to apply Policy transformation [v0, policy_simple]

            Return: 
                Image: A tensor of Applied transformation [with, height, channels]
        '''
        crp_shap = (crop_size, crop_size, 3)
        flp_img = tf.image.random_flip_left_right(image)
        rnd_flp_crp = tf.image.random_crop(flp_img, crp_shap)
        return rnd_flp_crp


    def rand_distribe_crop_global_local_views_flip(image, crop_size, min_scale, max_scale, high_resol=True):
        '''
            Args:
                image: A tensor [ with, height, channels]
                crop_size: Rand --> Flipping --> random_distribute_uniform (min_scale, max_scale) 
                high_resol --> True: For Global crop_view, False: For Local crop views
                AutoAugment: a function to apply AutoAugment transformation 

            Return: 
                Image: A tensor of Applied transformation [with, height, channels]
        '''
        flp_img = tf.image.random_flip_left_right(image)
        crp_ratio = crop_size * 1.4 if high_resol else crop_size * 0.8  
        image_shape = tf.cast(crp_ratio, dtype=tf.int32)
        image_shape = tf.cast(image_shape, tf.float32)
        resz_flp_img = tf.image.resize(flp_img, (image_shape, image_shape))

        size = tf.random.uniform(shape=(1,), minval=min_scale*image_shape, 
                        maxval=max_scale*image_shape, dtype=tf.float32)
        size = tf.cast(size, tf.int32)[0]
        # Get crop_size
        rnd_flp_crp = tf.image.random_crop(resz_flp_img, (size, size, 3))
        # Return image with Crop_size
        return tf.image.resize(rnd_flp_crp, (crop_size, crop_size))


    def data_augment(self, image, aug_type="default", crop_size=None, 
                min_scale=None, max_scale=None, high_resol=True):
        '''
            The common interface for public user, which allow user only call this method 
            to augument the given image with the corresponding behavior setting by args. 
            Args:
                image: A tensor [ with, height, channels]
                other args for various augumentation
            Return: 
                Image: A tensor of Applied transformation [with, height, channels]
        '''
        def distort(pre_img):
            if self.DAS_type == "iaa_rand_aug":
                return self.aug_inst(pre_img)
            return self.aug_inst.distort(pre_img)
        
        try:
            pre_img = self.pre_proc_dict[aug_type](image, crop_size, min_scale, 
                                    max_scale, high_resol)
        except KeyError as k_err:
            raise KeyError("Given vlaue {} of aug_type, \
            but the value should be one of that ({})".format(
                aug_type, self.pre_proc_dict.keys() )
            )
        except Exception as exc:
            print(exc) ; return

        aug_img = self.aug_inst.distort(pre_img)
        return self.post_proc(aug_img)


if __name__ == '__main__':
    # simple test
    import numpy as np

    da_inst = Data_Augumentor()
    dummy_img = np.random.random((3, 3, 4))
    aug_img = da_inst.data_augment(dummy_img)
    print(aug_img)