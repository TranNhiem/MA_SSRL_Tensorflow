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
    

def vis_data_aug(ds, batch_size=4):
    # perform batch before calling the map method benefit the training performance
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = multiview_wrap(ds)

    for mv_bh_img in ds:
        print(f"view size : { len(mv_bh_img) }\n")
        # output global view
        imshow_imgs(mv_bh_img[0].numpy(), "global_view examples")
        imshow_imgs(mv_bh_img[3].numpy(), "local_view examples")

        # perform un-mix strategy
        '''
        #   global view mixing
        glb_mix = mix_up_batch(mv_bh_img[0], mv_bh_img[1])
        #   local view mixing with pair
        loc_mix_23 = mix_up_batch(mv_bh_img[2], mv_bh_img[3])
        loc_mix_24 = mix_up_batch(mv_bh_img[2], mv_bh_img[4])
        imshow_imgs(mv_bh_img[0].numpy(), "global_view examples")
        imshow_imgs(mv_bh_img[3].numpy(), "local_view examples")
        '''

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