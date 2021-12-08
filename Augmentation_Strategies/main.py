# PEP8 : alphabet order 
import copy
from functools import partial 
from matplotlib import pyplot as plt
import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE  # global define var for ds.map parallel call

# self-def pkg
from Auto_Data_Augment.Data_Augmentor import Data_Augmentor
from Multi_Viewer.Multi_viewer import Multi_viewer
from View_Mixer.View_mixer import mix_up_batch
from vis_tool import imshow_imgs

def get_toy_dataset(dataset_name, split='train', shuffle_files=True, with_info=True):
    ds_lst = tfds.list_builders()
    assert dataset_name in ds_lst  
    # self-supervised without label
    ds, meta_info = tfds.load(dataset_name, split=split, shuffle_files=shuffle_files, 
                                with_info=with_info, as_supervised=False)
    ds_pkg = (ds, meta_info) if with_info else ds
    return ds_pkg


def multiview_wrap(ds, da_type="auto_aug"):
    # preproc for cifar10
    format_proc = lambda ex : ex["image"] # filter out the lable info
    ds = ds.map(format_proc, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # wrap the multi-view & data augment strategy
    da = Data_Augmentor(da_type)
    da.pre_proc_dict["default"] = lambda x : tf.cast(x, dtype=tf.float32) * 255.0 
    mv = Multi_viewer(da_inst=da)
    # wrap multi_view into tf.py_function, because of the leak of 'pseudo' auto-graph..
    out_typ_lst = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
    py_flow_wrap = lambda x : tf.py_function(mv.multi_view, [x], Tout=out_typ_lst)
    return ds.map(py_flow_wrap, num_parallel_calls=tf.data.experimental.AUTOTUNE)

   
def view_mixing_wrap(ds):

    def mix_view(*args):
        v01, l01 = mix_up_batch(args[0], args[1]) 
        v23, l23 = mix_up_batch(args[2], args[3])
        v24, l24 = mix_up_batch(args[2], args[4])
        return (v01, v23, v24), (l01, l23, l24)

    return ds.map(mix_view, num_parallel_calls=tf.data.experimental.AUTOTUNE)   
    

def vis_data_aug(ori_ds, batch_size=4):
    # perform batch before calling the map method benefit the training performance
    ds = ori_ds   # retain intermediate results
    ds = ds.batch(batch_size, drop_remainder=True)
    
    ## Kernel of the proposed method : multi-view & view mixing strategies 
    mv_ds = multiview_wrap(ds)
    mix_ds = view_mixing_wrap(mv_ds)
    
    # for loop-style solution (pass the test)
    for  bh_mix_img, lam in mix_ds.take(1):
        print(f"multiple view size : { len(bh_mix_img) }\n")
        # visualized example..
        imshow_imgs( bh_mix_img[0].numpy() , "global view mixed examples")
        imshow_imgs( bh_mix_img[1].numpy() , "local mixed 2, 3 view examples")
        imshow_imgs( bh_mix_img[2].numpy() , "local mixed 2, 4 view examples")
    

if __name__ == "__main__":
    cifar_ds, info = get_toy_dataset("cifar10")
    vis_data_aug(cifar_ds)