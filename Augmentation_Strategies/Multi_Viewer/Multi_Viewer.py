__author__ = "Rick & Josef"
__date__ = "2021/12/24"

from .Simclr_Byol_augmentation import distorted_bounding_box_crop
# use the unified cropping function
from .Byol_simclr_multi_croping_augmentation import simclr_augment_randcrop_global_views, \
                                                    simclr_augment_inception_style, supervised_augment_eval
from collections import namedtuple
import tensorflow as tf
import numpy as np


class Multi_viewer(object):
    View_spec = namedtuple("view_spec", "n_crp, re_siz, viw_siz, min_scale, max_scale")
    two_view = { "glb":View_spec(n_crp=2, re_siz=260, viw_siz=224, min_scale=0.5, max_scale=1),
                 "loc":View_spec(n_crp=3, re_siz=160, viw_siz=96, min_scale=0.14, max_scale=0.5) }
    def_da = lambda im : im

    def __init__(self, multi_view_spec=None, da_inst=None):
        self.multi_view_spec = multi_view_spec if multi_view_spec \
                            else Multi_viewer.two_view
        self.util = {'cnvt_typ':lambda x : tf.image.convert_image_dtype(x, tf.float32),
                        'cast':lambda x : tf.cast(x, tf.int32)[0],
                        'incpt_crp':lambda x, **_ : self.__inception_style_crop(x),                            # simclr_augment_inception_style(x), # 
                        'rnd_crp':lambda x, **args : self.__random_resize_crop(x, **args) }                    # simclr_augment_randcrop_global_views(x, **args) }# 
        self.da_inst = da_inst if da_inst else Multi_viewer.def_da

    # for global view only : inception style cropping
    def __inception_style_crop(self, image):
        batch_siz, height, width, channel = image.shape
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        aspect_ratio = width / height
        image = distorted_bounding_box_crop(
            image,
            bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(3. / 4. * aspect_ratio, 4. / 3. * aspect_ratio),
            area_range=(0.08, 1.0),
            max_attempts=100,
            scope=None)
        return tf.image.resize([image], [height, width],
                            method=tf.image.ResizeMethod.BICUBIC)[0]

    # for local and global view
    def __random_resize_crop(self, image, re_siz, viw_siz, min_scale, max_scale):
        image = tf.image.resize(image, (re_siz, re_siz))
        # get the crop size for given min and max scale
        crp_siz = tf.random.uniform(shape=(1,), minval=min_scale*re_siz, 
                                    maxval=max_scale*re_siz, dtype=tf.float32)
        crp_siz = self.util['cast'](crp_siz)
        # get the crop from the rgb image
        crp_im = tf.image.random_crop( image, (crp_siz, crp_siz, 3) )
        im_view = tf.image.resize(crp_im, (viw_siz, viw_siz))
        return im_view

    
    def multi_view(self, batch_image, y, da_type=None, incpt_crp=False):
        bth_im = self.util['cnvt_typ'](batch_image)
        bth_im_buff = []
        for viw_name, vs in self.multi_view_spec.items():
            crp_key = "incpt_crp" if viw_name == "glb" and incpt_crp \
                                else "rnd_crp"
            for num in range(vs.n_crp):
                im_buff = []
                kwargs = {"re_siz":vs.re_siz, "viw_siz":vs.viw_siz, 
                        "min_scale":vs.min_scale, "max_scale":vs.max_scale}
                bth_im = self.util[crp_key](bth_im, **kwargs)
                    
                # data augment perform batch image transformations
                #bth_im_buff.append( self.da_inst.data_augment(bth_im) )
                trfs_im = self.da_inst(bth_im)
                print(f"img shape : {trfs_im.shape}\n\n")
                
                if da_type == "fastaa":
                    bth_im_buff.append( trfs_im )  # now da_inst is just a function
                else:
                    bth_im_buff.append( [trfs_im] )  # now da_inst is just a function
                bth_im_buff.append( [y] )

        return bth_im_buff