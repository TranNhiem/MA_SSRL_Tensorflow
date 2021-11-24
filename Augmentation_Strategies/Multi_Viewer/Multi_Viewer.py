# Apply the mock flag
#from utility.absl_mock import flags
#FLAGS = flags.FLAGS

class Multi_viewer(object):
    two_view = {"glb":(260, 224), "loc":(160, 96)}
    
    def __init__(self, view_spec_dict=None):
        self.view_spec = view_spec_dict if view_spec_dict \
                            else Multi_viewer.two_view
        self.util = {'cnvt_typ':lambda x : tf.image.convert_image_dtype(x, tf.float32),
                        'cast':lambda x : tf.cast(x, tf.int32)[0] }

    # color distortion
    def __custom_augment(self, image):
        # Random flips
        image = random_apply(tf.image.flip_left_right, image, p=0.5)
        # Randomly apply gausian blur
        image = random_apply(gaussian_blur, image, p=0.5)
        # Randomly apply transformation (color distortions) with probability p.
        image = random_apply(color_jitter, image, p=0.8)
        # Randomly apply grayscale
        image = random_apply(color_drop, image, p=0.2)
        

    def multi_view(self, image, label, view_spec={"glb":(260, 224), "loc":(160, 96)},
                min_scal_rng=[0.5, 0.14], max_scal_rng=[1., 0.5], db_mod=False):
        im_lst = []
        image = self.util['cnvt_typ'](image)
        for view_name, (re_siz, viw_siz) in self.view_spec.items():
            image = tf.image.resize(image, (re_siz, re_siz))
            # get the crop size for given min and max scale
            crp_siz = tf.random.uniform(shape=(1,), minval=min_scale*re_siz, maxval=max_scale*re_siz, dtype=tf.float32)
            crp_siz = self.util['cast'](crp_siz)
            # get the crop from the image
            crp_shp = (crp_siz, crp_siz, 3)
            crp_im = tf.image.random_crop(image, crp_shp)
            im_view = tf.image.resize(crp_im, (viw_siz, viw_siz))
            distored_image = self.__custom_augment(im_view)

        return distored_image, label  # bypass label 
    