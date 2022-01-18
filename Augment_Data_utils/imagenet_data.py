'''
This Implementaion three Different Augmentation Strategy for ImageNet Dataset
1. Baseline - SimCLR Augmentation 
2. RandAug - RandAug Augmentation (Original and Modify)
3. AutoAugment -- Auto Augmentation Policies 

'''


import numpy as np
import tensorflow as tf
# Augmentation Policy
from imutils import paths
import os
import imgaug.augmenters as iaa
from Data_augmentation_policy.Simclr_augment_random import custom_augment
from Data_augmentation_policy.RandAugment import tfa_randaug, tfa_randaug_rand_crop_fliping, tfa_randaug_rand_ditris_uniform_croping
from Data_augmentation_policy.Auto_Augment import tfa_AutoAugment, tfa_AutoAugment_rand_crop_flip, tfa_AutoAugment_rand_distribe_crop_global_local_views_flip
from official.vision.image_classification.augment import RandAugment, AutoAugment

AUTO = tf.data.experimental.AUTOTUNE
SEED = 26


class imagenet_dataset():
    def __init__(self, IMG_SIZE, BATCH_SIZE, img_path):
        self.IMG_SIZE = IMG_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.x_train = list(paths.list_images(img_path))

    @classmethod
    def parse_images(self, image_path):
        # Loading and reading Image
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        #img=tf.image.convert_image_dtype(img, tf.float32)

        return img

    @classmethod
    def parse_images_label(self, image_path):
        img = tf.io.read_file(image_path)
        # img = tf.image.decode_jpeg(img, channels=3) # decode the image back to proper format
        img = tf.io.decode_jpeg(img, channels=3)
        label = tf.strings.split(image_path, os.path.sep)[3]
        # print(label)
        return img, label

    def ssl_Simclr_Augment_policy(self):

        train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                        .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                        .map(self.parse_images,  num_parallel_calls=AUTO)
                        .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                             num_parallel_calls=AUTO,
                             )
                        .map(lambda x: (custom_augment(x, self.IMG_SIZE)), num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                        .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                        .map(self.parse_images,  num_parallel_calls=AUTO)
                        .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                             num_parallel_calls=AUTO,
                             )
                        .map(lambda x: (custom_augment(x, self.IMG_SIZE)), num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        return train_ds

    def ssl_Randaug_Augment_IMGAUG_policy(self, num_transform, magnitude, crop_size, min_scale, max_scale, high_resol=True,  mode="original"):

        if crop_size is None:
            raise ValueError("you input invalide crop_size")

        rand_aug_apply = iaa.RandAugment(n=num_transform, m=magnitude)

        def imgaug_randaug(images):
            '''
            Args:
            images: A batch tensor [batch, with, height, channels]
            rand_aug: a function to apply Random transformation 
            Return: 
            Images: A batch of Applied transformation [batch, with, height, channels]
            '''

            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.
            images = tf.cast(images, tf.uint8)
            images = rand_aug_apply(images=images.numpy())
            #images = (images.astype(np.float32))/255.
            images = tf.cast(images, tf.float32)/255.
            return images

        def flip_random_crop(image, crop_size):
            '''
            Args: 
            image: tensor shape of [height, width, channels]
            crop_size: using for random crop 
            Return: 
            A tensor transform with Flipping and Crop same_size as image if Crop==img_size
            '''
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_crop(image, (crop_size, crop_size, 3))

            return image

        def rand_flip_crop_global_local_view(image, min_scale, max_scale, crop_size, high_resol=True):

            image = tf.image.random_flip_left_right(image)
            if high_resol:
                image_shape = tf.cast((crop_size * 1.4), dtype=tf.int32)
                image_shape = tf.cast(image_shape, tf.float32)
                # print(image_shape)
                image = tf.image.resize(image, (image_shape, image_shape))
            else:
                image_shape = tf.cast(crop_size * 0.8, dtype=tf.int32)
                image_shape = tf.cast(image_shape, tf.float32)
                # print(image_shape)
                image = tf.image.resize(image, (image_shape, image_shape))
            size = tf.random.uniform(shape=(
                1,), minval=min_scale*image_shape, maxval=max_scale*image_shape, dtype=tf.float32)
            size = tf.cast(size, tf.int32)[0]
            # Get crop_size
            crop = tf.image.random_crop(image, (size, size, 3))
            # Return image with Crop_size
            image = tf.image.resize(crop, (crop_size, crop_size))

        if mode == "orginal":
            print(" You Implement Imgaug Original")

            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .batch(self.BATCH_SIZE)
                            # .map(lambda x: (imgaug_randaug(x, num_transform=num_transform, magnitude=magnitude)), num_parallel_calls=AUTO)
                            .map(lambda x: (tf.py_function(imgaug_randaug,  [x], [tf.float32])[0]), num_parallel_calls=AUTO)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .batch(self.BATCH_SIZE)
                            # .map(lambda x: (imgaug_randaug(x, num_transform, magnitude)), num_parallel_calls=AUTO)
                            .map(lambda x: (tf.py_function(imgaug_randaug,  [x], [tf.float32])[0]), num_parallel_calls=AUTO)
                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        elif mode == "crop":

            print(" You implement Croping with ImgAug")

            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)

                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (flip_random_crop(x, crop_size)),  num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            # .map(lambda x: (imgaug_randaug(x, num_transform=num_transform, magnitude=magnitude)), num_parallel_calls=AUTO)
                            .map(lambda x: (tf.py_function(imgaug_randaug,  [x], [tf.float32])[0]), num_parallel_calls=AUTO)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (flip_random_crop(x, crop_size)),  num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            # .map(lambda x: (imgaug_randaug(x, num_transform, magnitude)), num_parallel_calls=AUTO)
                            .map(lambda x: (tf.py_function(imgaug_randaug,  [x], [tf.float32])[0]), num_parallel_calls=AUTO)
                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        elif mode == "global_local_crop":

            print("You implement Global and Local Crop View")

            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)

                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (rand_flip_crop_global_local_view(x, min_scale, max_scale, crop_size, high_resol=high_resol)),  num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            # .map(lambda x: (imgaug_randaug(x, num_transform=num_transform, magnitude=magnitude)), num_parallel_calls=AUTO)
                            .map(lambda x: (tf.py_function(imgaug_randaug,  [x], [tf.float32])[0]), num_parallel_calls=AUTO)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (rand_flip_crop_global_local_view(x, min_scale, max_scale, crop_size, high_resol=high_resol)),  num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            # .map(lambda x: (imgaug_randaug(x, num_transform, magnitude)), num_parallel_calls=AUTO)
                            .map(lambda x: (tf.py_function(imgaug_randaug,  [x], [tf.float32])[0]), num_parallel_calls=AUTO)
                            .prefetch(AUTO)
                            )

        return train_ds

    def ssl_Auto_Augment_TFA_policy(self, crop_size, min_scale, max_scale, high_resol=True, mode="original"):

        if crop_size is None:
            raise ValueError("you enter invalid crop_size")
        #mode ["original", "crop", "global_local_crop"]
        if high_resol:
            print("You Implement the Global Views")
        else:
            print("you implement local views")

        if mode == "original":
            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_AutoAugment(x,)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train))
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_AutoAugment(x,)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        elif mode == "crop":

            print("implement AutoAugment Rand Croping Fliping")
            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_AutoAugment_rand_crop_flip(x,  crop_size)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_AutoAugment_rand_crop_flip(x, crop_size)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        elif mode == "global_local_crop":

            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                              .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                  num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_AutoAugment_rand_distribe_crop_global_local_views_flip(x,  crop_size, min_scale, max_scale, high_resol=high_resol)), num_parallel_calls=AUTO)
                            # .map(lambda x: (tfa_AutoAugment_rand_crop_flip(x,  crop_size)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                              .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                  num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_AutoAugment_rand_distribe_crop_global_local_views_flip(x, crop_size, min_scale, max_scale, high_resol=high_resol)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        else:
            raise ValueError("Implementation mode is node in design")

        return train_ds

    def ssl_RandAugment_TFA_policy(self, num_transform, magnitude, crop_size, min_scale, max_scale, high_resol=True, mode="original"):

        if crop_size is None:
            raise ValueError("you enter invalid crop_size")
        #mode ["original", "crop", "global_local_crop"]
        if high_resol:
            print("You Implement the Global Views")
        else:
            print("you implement local views")

        if mode == "original":
            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_randaug(x, num_transform, magnitude,)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_randaug(x, num_transform, magnitude,)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        elif mode == "crop":

            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_randaug_rand_crop_fliping(x, num_transform, magnitude, crop_size)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_randaug_rand_crop_fliping(x, num_transform, magnitude, crop_size)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        elif mode == "global_local_crop":

            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_randaug_rand_ditris_uniform_croping(x, num_transform, magnitude,  crop_size, min_scale, max_scale, high_resol=high_resol)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_randaug_rand_ditris_uniform_croping(x, num_transform, magnitude, crop_size, min_scale, max_scale, high_resol=high_resol)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        else:
            raise ValueError("Implementation mode is node in design")

        return train_ds


# old code..

# Imagenet_dataset is a 'class', we don't do FLAGS in it, we use 'self'.
class Imagenet_dataset(object):
    # The cropping strategy can be applied
    crop_dict = {"incpt_crp": simclr_augment_inception_style,
                 "rnd_crp": simclr_augment_randcrop_global_views}
    # The crop strategy is 'incpt_style' in global view
    default_view = {"glb": Multi_viewer.View_spec(
        n_crp=1, re_siz=224, viw_siz=224, min_scale=0.5, max_scale=1)}

    def __init__(self, img_size, train_batch, val_batch, train_path=None, train_label=None,
                 val_path=None, val_label=None, strategy=None, subset_class_num=None, seed=None):
        '''
        Args: 
            img_size: Image training size
            train_batch: Distributed Batch_size for training multi-GPUs

            image_path: Directory to train data 
            val_path:   Directory to validation or testing data
            subset_class_num: subset class 
        '''
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = train_batch
        self.val_batch = val_batch
        self.strategy = strategy
        self.seed = seed

        self.label, self.class_name = self.get_label(train_label)
        numeric_train_cls = []
        numeric_val_cls = []
        print("train_path:", train_path)
        print("val_path:", val_path)

        if train_path is None and val_path is None:
            raise ValueError(
                f'The train_path and val_path is None, please cheeek')
        elif val_path is None:
            dataset = list(paths.list_images(train_path))
            dataset_len = len(dataset)
            random.Random(self.seed).shuffle(dataset)
            self.x_val = dataset[0:int(dataset_len * 0.2)]
            self.x_train = dataset[len(self.x_val) + 1:]
            for image_path in self.x_train:
                label = re.split(r"/|\|//|\\", image_path)[-2]
                #label = image_path.split("/")[-2]
                numeric_train_cls.append(self.label[label])
            for image_path in self.x_val:
                label = re.split(r"/|\|//|\\", image_path)[-2]
                numeric_val_cls.append(self.label[label])

        else:
            self.x_train = list(paths.list_images(train_path))

            self.x_val = list(paths.list_images(val_path))
            random.Random(self.seed).shuffle(self.x_train)
            random.Random(self.seed).shuffle(self.x_val)

            for image_path in self.x_train:
                label = re.split(r"/|\|//|\\", image_path)[-2]
                numeric_train_cls.append(self.label[label])

            val_label_map = self.get_val_label(val_label)
            numeric_val_cls = []
            for image_path in self.x_val:
                label = re.split(r"/|\|//|\\", image_path)[-1]

                label = label.split("_")[-1]
                label = int(label.split(".")[0])
                numeric_val_cls.append(val_label_map[label-1])

        if subset_class_num != None:
            x_train_sub = []
            numeric_train_cls_sub = []
            for file_path, numeric_cls in zip(self.x_train, numeric_train_cls):
                if numeric_cls < subset_class_num:
                    x_train_sub.append(file_path)
                    numeric_train_cls_sub.append(numeric_cls)
            self.x_train = x_train_sub
            numeric_train_cls = numeric_train_cls_sub

            x_val_sub = []
            numeric_val_cls_sub = []
            for file_path, numeric_cls in zip(self.x_val, numeric_val_cls):
                if numeric_cls < subset_class_num:
                    x_val_sub.append(file_path)
                    numeric_val_cls_sub.append(numeric_cls)
            self.x_val = x_val_sub
            numeric_val_cls = numeric_val_cls_sub

        # Path for loading all Images
        # For training
        self.x_train_lable = tf.one_hot(numeric_train_cls, depth=len(
            self.class_name) if subset_class_num == None else subset_class_num)
        self.x_val_lable = tf.one_hot(numeric_val_cls, depth=len(
            self.class_name) if subset_class_num == None else subset_class_num)

    def get_label(self, label_txt_path=None):
        class_name = []
        class_ID = []
        class_number = []
        print(label_txt_path)
        with open(label_txt_path) as file:
            for line in file.readlines():
                # n02119789 1 kit_fox
                lint_split = line.split(" ")
                class_ID.append(lint_split[0])
                class_number.append(int(lint_split[1]))
                class_name.append(lint_split[2])
            file.close()

        label = dict(zip(class_ID, class_number))
        class_name = dict(zip(class_ID, class_name))
        return label, class_name

    def get_val_label(self, label_txt_path=None):
        class_number = []
        with open(label_txt_path) as file:
            for line in file.readlines():
                class_number.append(int(line[:-1]))
                # n02119789 1 kit_fox
        return class_number

    def __parse_images_lable_pair(self, image_path, label):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, label

    def __wrap_ds(self, img_folder, labels):
        img_lab_ds = tf.data.Dataset.from_tensor_slices((img_folder, labels)) \
            .shuffle(self.BATCH_SIZE * 100, seed=self.seed) \
            .map(lambda x, y: (self.__parse_images_lable_pair(x, y)), num_parallel_calls=AUTO)\
            .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y), num_parallel_calls=AUTO).cache()
        return img_lab_ds

    def __wrap_da(self, ds, trfs, wrap_type="cropping"):
        if wrap_type == "cropping":
            def map_func(x, y): return (trfs(x, self.IMG_SIZE), y)

        elif wrap_type == "validate":
            def map_func(x, y): return (trfs(x, FLAGS.IMG_height, FLAGS.IMG_width,
                                             FLAGS.randaug_transform, FLAGS.randaug_magnitude), y)
        elif wrap_type == "data_aug":  # careful, tf.py_func ret_val include dummy-dim, so we unzip * it
            def map_func(x, y): return (
                *tf.py_function(trfs, [x], Tout=[tf.float32]), y)

        else:  # ignore the label to simplify mixing view implementation
            def map_func(x, y): return tf.py_function(trfs, [x], Tout=[
                tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

        logging.info("applying resize in wrap_ds for Caching Implementation")
        # apply DA only
        data_aug_ds = ds.map(map_func, num_parallel_calls=AUTO) \
            .batch(self.BATCH_SIZE, num_parallel_calls=AUTO) \
            .prefetch(20)
        return data_aug_ds

    # This for Supervised validation training
    def supervised_validation(self):
        raw_ds = self.__wrap_ds(self.x_train, self.x_train_lable)
        val_ds = self.__wrap_da(raw_ds, supervised_augment_eval, "validate")
        logging.info("Val_ds with option")
        val_ds.with_options(options)

        return self.strategy.experimental_distribute_dataset(val_ds)

    def simclr_crop_da(self, crop_type="incpt_crp"):
        if not crop_type in Imagenet_dataset.crop_dict.keys():
            raise ValueError(
                f"The given cropping strategy {crop_type} is not supported")

        ds_one = self.__wrap_ds(self.x_train, self.x_train_lable)
        train_ds_one = self.__wrap_da(ds_one, self.crop_dict[crop_type])

        ds_two = self.__wrap_ds(self.x_train, self.x_train_lable)
        train_ds_two = self.__wrap_da(ds_two, self.crop_dict[crop_type])

        if FLAGS.dataloader == "ds_1_2_options":
            logging.info("Train_ds_one and two  with option")
            train_ds_one.with_options(options)
            train_ds_two.with_options(options)

        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        if FLAGS.dataloader == "train_ds_options":
            logging.info("Train_ds dataloader with option")
            train_ds.with_options(options)
        # else:
        #     logging.info(" dataloader without option")
        #train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        return self.strategy.experimental_distribute_dataset(train_ds)

    def auto_data_aug(self, da_type="auto_aug", crop_type="incpt_crp", *aug_args, **aug_kwarg):
        da_inst = Data_Augmentor(DAS_type=da_type, *aug_args, **aug_kwarg) if da_type \
            else Data_Augmentor(*aug_args, **aug_kwarg)

        ds_one = self.__wrap_ds(self.x_train, self.x_train_lable)
        ds_one = ds_one.map(lambda x, y: (self.crop_dict[crop_type](
            x, self.IMG_SIZE), y), num_parallel_calls=AUTO)
        train_ds_one = self.__wrap_da(ds_one, da_inst.data_augment, "data_aug")

        ds_two = self.__wrap_ds(self.x_train, self.x_train_lable)
        ds_two = ds_two.map(lambda x, y: (self.crop_dict[crop_type](
            x, self.IMG_SIZE), y), num_parallel_calls=AUTO)
        train_ds_two = self.__wrap_da(ds_two, da_inst.data_augment, "data_aug")

        if FLAGS.dataloader == "ds_1_2_options":
            logging.info("Train_ds_one and two  with option")
            train_ds_one.with_options(options)
            train_ds_two.with_options(options)

        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        if FLAGS.dataloader == "train_ds_options":
            logging.info("Train_ds dataloader with option")
            train_ds.with_options(options)

        # else:
        #     logging.info(" dataloader without option")

        #train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        return self.strategy.experimental_distribute_dataset(train_ds)

    def multi_view_data_aug(self, da_type="auto_aug"):
        da = Data_Augmentor(DAS_type=da_type) if da_type else Data_Augmentor()
        mv = Multi_viewer(da_inst=da)

        raw_ds = self.__wrap_ds(self.x_train, self.x_train_lable)
        tra_ds_lst = self.__wrap_da(raw_ds,  mv.multi_view, "mv_aug")
        train_ds = tf.data.Dataset.zip(tra_ds_lst)

        logging.info("Train_ds_multiview dataloader with option")
        train_ds.with_options(options)

        return self.strategy.experimental_distribute_dataset(train_ds)

    def get_data_size(self):
        return len(self.x_train), len(self.x_val)
