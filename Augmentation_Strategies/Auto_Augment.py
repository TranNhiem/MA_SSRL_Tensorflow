__src_str__ = '''
  Auto Augmentation Policy Focus on Object Detection Task 
  #Reference Implementation on Object Detection with 
  1. AutoAugment 3 Policies (V0- V3)
  Barret, et al. Learning Data Augmentation Strategies for Object Detection.
      Arxiv: https://arxiv.org/abs/1906.11172
  2. RandomAugment --> Also apply for object Detection Models

  ## Reference GitHub for implementation
  [1] https://github.com/google/automl/blob/master/efficientdet/aug/autoaugment.py
  [2] https://github.com/tensorflow/models/blob/master/official/vision/image_classification/augment.py
'''

# Current implementation will Deploy for Images WITHOUT BOX
import tensorflow as tf
from tensorflow.image import random_flip_left_right, random_crop
from official.vision.image_classification.augment import AutoAugment

class Data_Augumentor(object):
    # static instance, declare once use everywhere!
    aug_inst = AutoAugment(augmentation_name='v0')
    def __init__(self):  
        self.pre_proc_dict = {"default":lambda image, *_ : tf.cast(image, dtype=tf.float32),
                    "rnd_crp_flp":self.rand_crop_flip,
                    "glb_loc_crp_flp":self.rand_distribe_crop_global_local_views_flip}
        self.post_proc = lambda image : tf.cast(image, dtype=tf.float32) / 255.


    @staticmethod
    def prnt_policies():
        print("AutoAugment Policy V0-- implementation : \n")
        print("V0--> policy = \n")
        print(Data_Augumentor.aug_inst.policy_v0())
        print("\n\n=================\n\n")
        print("Policy_simple = \n")
        print(Data_Augumentor.aug_inst.policy_simple())

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
        try:
            pre_img = self.pre_proc_dict[aug_type](image, crop_size, min_scale, 
                                    max_scale, high_resol)
        except KeyError as k_err:
            raise KeyError("Given vlaue {} of aug_type, \
            but the value should be one of that ({})".format(
                aug_type, ["default", "rnd_crp_flp", "glb_loc_crp_flp"]
            ))
        except Exception as exc:
            print(exc)

        aug_img = Data_Augumentor.aug_inst.distort(pre_img)
        return self.post_proc(aug_img)

if __name__ == '__main__':
    # simple test
    import numpy as np

    da_inst = Data_Augumentor()
    dummy_img = np.random.random((3, 3, 4))
    aug_img = da_inst.data_augment(dummy_img)
    print(aug_img)

