import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from vis_tool import imshow_imgs
from matplotlib import pyplot as plt
from functools import partial 

# self-def pkg
from Auto_Data_Augment.Data_Augmentor import Data_Augmentor
from Multi_Viewer.Multi_viewer import Multi_viewer
from View_Mixer.View_mixer import mix_up_batch

def vis_data_aug(ds, da_type="auto_aug", batch_size=4):
    da = Data_Augmentor(da_type)
    da.pre_proc_dict["default"] = lambda x : tf.cast(x, dtype=tf.float32) * 255.0 
    mv = Multi_viewer(da_inst=da)

    ## Multi-view + Un-mix strategy un-loop style:
    for example in ds.batch(batch_size, drop_remainder=True):
        bh_img, lab = example["image"], example["label"]
        imshow_imgs( bh_img.numpy() )
        mv_bh_img = mv.multi_view(bh_img)  
        print(f"view size : { len(mv_bh_img) }\n")
        # output global view
        imshow_imgs(mv_bh_img[0], "global_view examples")
        imshow_imgs(mv_bh_img[3], "local_view examples")
        # global view mixing
        glb_mix = mix_up_batch(mv_bh_img[0], mv_bh_img[1])
        # local view mixing with pair
        loc_mix_23 = mix_up_batch(mv_bh_img[2], mv_bh_img[3])
        loc_mix_24 = mix_up_batch(mv_bh_img[2], mv_bh_img[4])
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