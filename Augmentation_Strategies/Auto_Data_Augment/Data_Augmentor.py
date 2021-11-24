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
from .Fast_Auto_Augment import Fast_AutoAugment
# tf-models-official 
from .tf_official_DA import AutoAugment, RandAugment

# For 'Type Hint' function, we replace the comment of code into type package.. 
#from typing import Any, Dict, List, Optional, Text, Tuple
from functools import partial
import numpy as np

class Data_Augmentor(object):
    # static instance, declare once use everywhere!
    DAS_dict = {"auto_aug":AutoAugment, "rand_aug":RandAugment, "fast_aug":Fast_AutoAugment, "simCLR":None}
    # common method name to apply the data augmentation!!
    DA_METHOD = "distort"
    def_preproc = lambda image : tf.cast(image, dtype=tf.float32)
    def_postproc = lambda image : tf.cast(image, dtype=tf.float32) / 255.

    def __init__(self, DAS_type="auto_aug", *aug_args, **aug_kwarg):
        try:
            self.DAS_type = DAS_type
            self.aug_inst = Data_Augmentor.DAS_dict[DAS_type](*aug_args, **aug_kwarg)
        except KeyError as k_err:
            raise KeyError("Given vlaue {} of DAS_type, \
                            but the value should be one of that ({})".format(
                            aug_type, Data_Augmentor.DAS_dict.keys() )
                          )
        except Exception as exc:
            print(exc)

        self.pre_proc_dict = {"default":Data_Augmentor.def_preproc}
        self.post_proc_dict = {"default":Data_Augmentor.def_postproc}
        self.regist_common_distort()


    @staticmethod
    def prnt_policies(DAS_type):
        # please refer the DAS_dict keys
        da_lst = Data_Augmentor.DAS_dict.keys()
        if DAS_type == da_lst[0]:
            print("AutoAugment Policy V0-- implementation : \n")
            print("V0--> policy = \n")
            print(Data_Augmentor.DAS_dict[DAS_type].policy_v0())
            print("\n\n=================\n\n")
            print("Policy_simple = \n")
            print(Data_Augmentor.DAS_dict[DAS_type].policy_simple())

        elif DAS_type == da_lst[1]:
            print("RandAugment avaliable ops : \n")
            print(Data_Augmentor.DAS_dict[DAS_type].available_ops)

        elif DAS_type == da_lst[2]: 
            print("FastAutoAugment avaliable ops : \n")
            print(Data_Augmentor.DAS_dict[DAS_type]().prnt_policies)
            
        else:
            raise ValueError("Given vlaue {} of DAS_type, \
                            but the value should be one of that ({})".format(
                            aug_type, Data_Augmentor.DAS_dict.keys() )
                          )


    # allow the augument instance have common method name to apply the data transformation
    def regist_common_distort(self):
        # the distort method already implemented in augument instance
        if Data_Augmentor.DA_METHOD in dir(self.aug_inst):
            return
        # plz regist the correct method, which apply the data transformation.. 
        

    def data_augment(self, image, preproc_lst=[], postproc_lst=[], db_mod=False):
        '''
            The common interface for public user, which allow user only call this method 
            to augument the given image with the corresponding behavior setting by args. 
            Args:
                image: A tensor [ with, height, channels]
                db_mod : Boolean, turn debug mode on/off (True/False) 
                other args will be removed in next commit
            Return: 
                Image: A tensor of Applied transformation [with, height, channels]
        '''
        ## HACKME : distort function can be designed in parallel pattern to improve the performance..
        def _distort(img_tnsr_lst):
            img_lst = []
            for img_tnsr in img_tnsr_lst:
                aug_img, trfs_lst = self.aug_inst.distort(img_tnsr)
                img_lst.append(aug_img)
                if db_mod:
                    print(trfs_lst)
            return img_lst

        def _img_proc(img_tnsr_lst, proc_func):
            img_lst = []
            for img_tnsr in img_tnsr_lst:
                img_lst.append( proc_func(img_tnsr) )
            return img_lst

        # 1. image pre-processing 
        try:
            pre_img = image
            for proc_typ in preproc_lst:
                preproc = self.pre_proc_dict[proc_typ]
                pre_img = _img_proc(pre_img, preproc)
        except KeyError as k_err:
            raise KeyError("Given preprocess type {} has not be registered in the pre_proc_dict, \
            the registered preprocess type : ({})".format(
                proc_typ, self.pre_proc_dict.keys() )
            )
        except Exception as exc:
            raise Exception(exc)
        
        # 2. apply the data augmentation of the image
        aug_img = _distort(pre_img)

        # 3. image post-processing 
        try:
            post_img = aug_img
            for proc_typ in postproc_lst:
                postproc = self.post_proc_dict[proc_typ]
                post_img = _img_proc(post_img, postproc)
        except KeyError as k_err:
            raise KeyError("Given postprocess type {} has not be registered in the post_proc_dict, \
            the registered postprocess type : ({})".format(
                proc_typ, self.post_proc_dict.keys() )
            )
        except Exception as exc:
            raise Exception(exc)
        # convert numpy dtype into the tf.Tensor for tf.Model processing..
        tf_cnvt = lambda img_lst : tf.convert_to_tensor(img_lst, dtype=tf.float32)
        post_img = tf_cnvt(post_img)

        return post_img


if __name__ == '__main__':
    # simple test
    import numpy as np

    da_inst = Data_Augmentor()
    dummy_img = np.random.random((3, 3, 4))
    aug_img = da_inst.data_augment(dummy_img)
    print(aug_img)