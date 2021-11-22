import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
#from Auto_Data_Augment.test.test_Data_Augmentor import Test_Data_Augumentor
from Auto_Data_Augment.Data_Augmentor import Data_Augmentor
import tensorflow_datasets as tfds
import tensorflow as tf
from vis_tool import imshow_imgs
from matplotlib import pyplot as plt

# import global / local view function
from functools import partial 
from Simclr_Byol_augmentation import (flip_random_crop, random_crop_flip_resize, 
                    inception_style_croping, rand_distribe_crop_global_local_views_flip)

def vis_data_aug(ds, da_type="auto_aug", batch_size=4):
    da = Data_Augmentor(da_type)
    # wrap view function
    incpt_crp = partial(inception_style_croping, height=20, width=20)
    local_crp = partial(rand_distribe_crop_global_local_views_flip, min_scale=0.5, max_scale=0.5)
    da.pre_proc_dict["incpt_crp"] = incpt_crp
    da.pre_proc_dict["loc_crp"] = local_crp
    
    ds = ds.batch(batch_size, drop_remainder=True)
    for example in ds:
        bh_img, lab = example["image"], example["label"]
        imshow_imgs( bh_img.numpy() )
        da_bh_img = da.data_augment(bh_img, ["loc_crp"], db_mod=True)
        imshow_imgs( da_bh_img.numpy() )


def get_toy_dataset(dataset_name, split='train', shuffle_files=True, with_info=True):
    ds_lst = tfds.list_builders()
    assert dataset_name in ds_lst  
    # self-supervised without label
    ds, meta_info = tfds.load(dataset_name, split=split, shuffle_files=shuffle_files, 
                                with_info=with_info, as_supervised=False)
    ds_pkg = (ds, meta_info) if with_info else ds
    return ds_pkg


if __name__ == "__main__":
    cifar_ds, info = get_toy_dataset("cifar10")
    vis_data_aug(cifar_ds)
    #test_report()
    