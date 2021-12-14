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
    format_proc = lambda ex :  ex["image"] 
    ds = ds.map(format_proc, num_parallel_calls=AUTOTUNE)

    # wrap the multi-view & data augment strategy
    da = Data_Augmentor(da_type)
    da.pre_proc_dict["default"] = lambda x : tf.cast(x, dtype=tf.float32) * 255.0 
    mv = Multi_viewer(da_inst=da)
    # wrap multi_view into tf.py_function, because of the leak of 'pseudo' auto-graph..
    out_typ_lst = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
    py_flow_wrap = lambda x : tf.py_function(mv.multi_view, [x], Tout=out_typ_lst)
    return ds.map(py_flow_wrap, num_parallel_calls=AUTOTUNE)

   
def view_mixing_wrap(ds):
    ...
    

def vis_data_aug(ori_ds, batch_size=4):
    # perform batch before calling the map method benefit the training performance
    ds = ori_ds   # retain intermediate results
    ds = ds.batch(batch_size, drop_remainder=True)
    
    ## Kernel of the proposed method : multi-view & view mixing strategies 
    mv_ds = multiview_wrap(ds)
    
    for mv_bh_img in mv_ds:
        print(f"mixed view size : { len(mv_bh_img) }\n")
        v01 = mix_up_batch(mv_bh_img[0], mv_bh_img[1])
        v23 = mix_up_batch(mv_bh_img[2], mv_bh_img[3])
        v24 = mix_up_batch(mv_bh_img[2], mv_bh_img[4])

        imshow_imgs(v01.numpy(), "global view mixed examples")
        imshow_imgs(v23.numpy(), "local view mixed examples 1)")
        imshow_imgs(v24.numpy(), "local view mixed examples 2)")


    '''
    for bh_img, mv_bh_img, mix_bh_img in zip(ori_ds, mv_ds, mix_ds):
        ## shape infomation :
        print(f"origin batch size : { len(bh_img) }\n")
        print(f"multiple view size : { len(mv_bh_img) }\n")
        print(f"mixed view size : { len(mix_bh_img) }\n")
        ## visualized example :
        # show original batch img
        imshow_imgs(bh_img.numpy(), "origin view of examples")
        # default setting is 2 view (global/local)
        imshow_imgs(mv_bh_img[0].numpy(), "global view examples")
        imshow_imgs(mv_bh_img[3].numpy(), "local view examples")
        # default setting is 3 output mixed view 
        #   (mix<glb_1, glb_2>, mix<loc_1, loc_2>, mix<loc_1, loc_3>) 
        #       omit the mix<loc2, loc_3> to prevent the heavy computation cost..
        imshow_imgs(mv_bh_img[0].numpy(), "global view mixed examples")
        imshow_imgs(mv_bh_img[1].numpy(), "local view mixed examples 1)")
        imshow_imgs(mv_bh_img[2].numpy(), "local view mixed examples 2)")
    '''


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

    cifar_ds, info = get_toy_dataset("cifar10")
    vis_data_aug(cifar_ds)
