import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from vis_tool import imshow_imgs
from matplotlib import pyplot as plt

# import global / local view function
from functools import partial 
from Simclr_Byol_augmentation import (flip_random_crop, random_crop_flip_resize, 
                    inception_style_croping, rand_distribe_crop_global_local_views_flip)

# self-def pkg
from Auto_Data_Augment.Data_Augmentor import Data_Augmentor
from Multi_Viewer.Multi_viewer import Multi_viewer


def vis_data_aug(ds, da_type="auto_aug", batch_size=4):
    da = Data_Augmentor(da_type)
    da.pre_proc_dict["default"] = lambda x : tf.cast(x, dtype=tf.float32) * 255.0 
    mv = Multi_viewer(da_inst=da)

    for example in ds.batch(batch_size, drop_remainder=True):
        bh_img, lab = example["image"], example["label"]
        imshow_imgs( bh_img.numpy() )
        mv_bh_img = mv.multi_view(bh_img)  
        print(f"view size : { len(mv_bh_img) }\n")
        # output global view
        imshow_imgs(mv_bh_img[0], "global_view examples")
        imshow_imgs(mv_bh_img[3], "local_view examples")
        break


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
    
    