__author__ = "Rick & Josef (refactor)"
__date__ = "2021/01/18"
from .Byol_simclr_multi_croping_augmentation import simclr_augment_randcrop_global_views, \
    simclr_augment_inception_style, supervised_augment_eval, simclr_augment_style
from Augmentation_Strategies.Multi_Viewer.Multi_Viewer import Multi_viewer
from absl import logging
from imutils import paths
import numpy as np
import random
import re
import os

# from Augmentation_Strategies.Auto_Data_Augment.Data_Augmentor import Data_Augmentor
from Augmentation_Strategies.Auto_Data_Augment.tf_official_DA import AutoAugment as autoaug
from Augmentation_Strategies.Auto_Data_Augment.tf_official_DA import RandAugment, Proposed_RandAugment, Extend_RandAugment
from Augmentation_Strategies.Auto_Data_Augment.Fast_Auto_Augment.Fast_AutoAugment import Fast_AutoAugment

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
AUTO = tf.data.experimental.AUTOTUNE
# Experimental options
options = tf.data.Options()
#tf.data.experimental.DistributeOptions()
options.experimental_optimization.noop_elimination = True
# options.experimental_optimization.map_vectorization.enabled = True
options.experimental_optimization.map_and_batch_fusion = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.apply_default_optimizations = True
#options.experimental_deterministic = False
options.experimental_threading.max_intra_op_parallelism = 1
# Shard policy using multi-machines training
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO

from typing import List, Dict
import tensorflow_datasets as tfds

# Define meta-cfg for parallel training
from config.absl_mock import Mock_Flag
flag = Mock_Flag()
FLAGS = flag.FLAGS
FLAGS.mode_prefetch = 1

if FLAGS.mode_prefetch == 1:
    mode_prefetch = AUTO
else:
    mode_prefetch = FLAGS.mode_prefetch


class Imagenet_dataset(object):
    # The cropping strategy can be applied
    crop_dict = {"incpt_crp": simclr_augment_inception_style,
                 "rnd_crp": simclr_augment_randcrop_global_views}

    def __init__(self, img_size, train_batch, val_batch, train_path=None, train_label=None, tra_ds_ratio=None,
                 val_path=None, val_label=None, strategy=None, subset_class_num=None, subset_percentage=None, seed=None, n_cls=None):
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
        self.tra_ds_ratio = tra_ds_ratio
        self.n_cls = n_cls
        self.x_train = []
        self.x_val = []

        self.label, self.class_name = self.get_label(train_label)
        numeric_train_cls = []
        numeric_val_cls = []
        print("train_path:", train_path)
        print("val_path:", val_path)

        if train_path is None and val_path is None:
            raise ValueError(
                f'The train_path and val_path is None, please cheeek')
        elif val_path is None:
            # dataset = list(paths.list_images(train_path))
            # dataset_len = len(dataset)
            # random.Random(FLAGS.SEED_data_split).shuffle(dataset)
            # self.x_val = dataset[0:int(dataset_len * 0.2)]
            # self.x_train = dataset[len(self.x_val) + 1:]
            
            self.get_train_path(train_path,subset_percentage)
            dataset_len = len(self.x_train)
            self.x_val = self.x_train[0:int(dataset_len * 0.2)]
            self.x_train = self.x_train[len(self.x_val) + 1:]


            for image_path in self.x_train:
                label = re.split(r"/|\|//|\\", image_path)[-2]
                # label = image_path.split("/")[-2]
                numeric_train_cls.append(self.label[label])
            for image_path in self.x_val:
                label = re.split(r"/|\|//|\\", image_path)[-2]
                numeric_val_cls.append(self.label[label])

        else:
            # self.x_train = list(paths.list_images(train_path))
            self.get_train_path(train_path,subset_percentage)

            self.x_val = list(paths.list_images(val_path))
            random.Random(FLAGS.SEED_data_split).shuffle(self.x_train)
            random.Random(FLAGS.SEED_data_split).shuffle(self.x_val)

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
    
    def get_train_path(self,train_path,subset_percentage):
        dir_names = os.listdir(train_path)
        for dir_name in dir_names:
            full_path = os.path.join(train_path, dir_name)
            if os.path.isdir(full_path):
                class_image_path = list(paths.list_images(full_path))
                if subset_percentage != None and len(class_image_path) > 10:
                    class_image_path = class_image_path[0:int(len(class_image_path) * subset_percentage)]
                self.x_train += class_image_path

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
        def parse_images(image_path):
            # Loading and reading Image
            img = tf.io.read_file(image_path)
            img = tf.io.decode_jpeg(img, channels=3)
            # norm into [0, 1] automatically
            img = tf.image.convert_image_dtype(img, tf.float32)
            return img
        return parse_images(image_path), label

    ##############################################################################################
    ## Load data by tfds API, time_stamp : 2022/02/05, 12:27
    ## On of the easy way to prepare the necessary dataset is loading the data from tfds.
    def load_by_tfds(self, split_lst:List=['train', 'validation', 'test']) -> Dict:
        '''load the dataset from the tfds api
            split_lst : the susbset you want to load from dataset, or the ratio in each subset
        '''

        # It will return the tuple(img, lab) tf.dataset instance when as_supervised enable!
        #   for default split_lst : ['train', 'validation', 'test'] but you can customized!
        #   you can find the imagenet2012 ds under /data/tf_ds
        ds_lst = tfds.load('imagenet2012', data_dir='/data/tf_ds', download=True, as_supervised=True, split=split_lst)
        key_lst = []
        # the key of access each ds-instance, you will need it outside! 
        if len(split_lst) == 1:
            key_lst.append('train')
        elif len(split_lst) == 2:
            key_lst.append('train') ; key_lst.append('test')
        elif len(split_lst) == 3:
            key_lst.append('train') ; key_lst.append('validation') ; key_lst.append('test')
        else:
            raise NotImplementedError("For simplify implementation, we only support you split dataset into 3 subset at maximum : ('train', 'validation', 'test')")
        
        ds_dict = {}  
        for ds, ds_key in zip(ds_lst, key_lst):
            ds_dict[ds_key] = ds
        return ds_dict

    # public insterface :
    def get_imgnet_ds(self):
        # In brief, multi-view must do resize wo cache, two have a choice to do resize or not with cache
        def wrap_tfds(ds):
            # data_info record the path of imgs, it should be parsed
            img_shp = (self.IMG_SIZE, self.IMG_SIZE)
            ## Wrapping procedure : copy & paste from wrap_ds
            if FLAGS.training_loop =="two_views": 
                print("Two_Views Wrap_ds")
                if FLAGS.resize_wrap_ds:
                    img_lab_ds = ds.map(lambda x, y: (tf.image.resize(x, img_shp), y), num_parallel_calls=AUTO)#.cache()
                
                else:
                    img_lab_ds = ds

            elif FLAGS.training_loop =="multi_views": 
                print("Multi_Views Wrap_ds")
                img_lab_ds = ds.map(lambda x, y: (tf.image.resize(x, img_shp), y), num_parallel_calls=AUTO)#.cache()

            else: 
                raise ValueError("Invalid_Training loop")
            
            if self.n_cls == None:
                raise ValueError("should give num of pred class")

            return img_lab_ds.map(lambda x, y : (x, tf.one_hot(y, depth=self.n_cls)))

        split_lst = [f'train[0:{self.tra_ds_ratio}%]']
        ds_dict = self.load_by_tfds(split_lst)
        ds = wrap_tfds(ds_dict['train'])
        return ds
        # did val_ds, tst_ds alos need to resize ?
        #except_lst = [] # according to the code, there is not exception
        #for ds_key, ds in ds_dict.items():  # automatic version
        #    if ds_key not in except_lst:
        #        ds_dict[ds_key] = wrap_tfds(ds)

        # or you can directly return the ds-instance to prevent unpacking from dict, but careful about the order
        #tra_ds, val_ds, tst_ds = ds_dict['train'], ds_dict['validation'], ds_dict['test']
        #return # ds_dict['train']
    ##############################################################################################

    def wrap_ds(self, img_folder, labels):
        # data_info record the path of imgs, it should be parsed
        img_shp = (self.IMG_SIZE, self.IMG_SIZE)
        
        if FLAGS.training_loop =="two_views": 
            print("Two_Views Wrap_ds")
            if FLAGS.resize_wrap_ds:
                img_lab_ds = tf.data.Dataset.from_tensor_slices((img_folder, labels)) \
                    .map(lambda x, y: (self.__parse_images_lable_pair(x, y)), num_parallel_calls=AUTO)\
                    .map(lambda x, y: (tf.image.resize(x, img_shp), y), num_parallel_calls=AUTO).cache()

            else:
                img_lab_ds = tf.data.Dataset.from_tensor_slices((img_folder, labels)) \
                    .map(lambda x, y: (self.__parse_images_lable_pair(x, y)), num_parallel_calls=AUTO).cache()

        elif FLAGS.training_loop =="multi_views": 
            print("Multi_Views Wrap_ds")
            img_lab_ds = tf.data.Dataset.from_tensor_slices((img_folder, labels)) \
            .map(lambda x, y: (self.__parse_images_lable_pair(x, y)), num_parallel_calls=AUTO)\
            .map(lambda x, y: (tf.image.resize(x, img_shp), y), num_parallel_calls=AUTO)#.cache()

        else: 
            raise ValueError("Invalid_Training loop")
            

        return img_lab_ds

    def wrap_da(self, ds, trfs, wrap_type="cropping"):
        if wrap_type == "cropping":
            def map_func(x, y): return (trfs(x, self.IMG_SIZE), y)

        elif wrap_type == "validate":
            def map_func(x, y): return (trfs(x, FLAGS.IMG_height, FLAGS.IMG_width,
                                             FLAGS.randaug_transform, FLAGS.randaug_magnitude), y)
        elif wrap_type == "data_aug":  # careful, tf.py_func ret_val include dummy-dim, so we unzip * it
            def map_func(x, y): return (
                *tf.py_function(trfs, [x], Tout=[tf.float32]), y)

        else:  # ignore the label to simplify mixing view implementation
            def map_func(x, y): return (tf.py_function(trfs, [x], Tout=[
                tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]), y)

        if FLAGS.resize_wrap_ds:
            logging.info(
                "applying resize in wrap_ds for Caching Implementation")
            data_aug_ds = ds.map(map_func, num_parallel_calls=AUTO) \
                .batch(self.BATCH_SIZE, num_parallel_calls=AUTO) \
                .prefetch(mode_prefetch)  # AUTO

        else:
            img_shp = (self.IMG_SIZE, self.IMG_SIZE)
            data_aug_ds = ds.map(lambda x, y: (tf.image.resize(x, img_shp), y), num_parallel_calls=AUTO) \
                            .map(map_func, num_parallel_calls=AUTO) \
                .batch(self.BATCH_SIZE, num_parallel_calls=AUTO) \
                .prefetch(mode_prefetch)  # AUTO

        return data_aug_ds

    # This for Supervised validation training
    def supervised_validation(self):
        raw_ds = self.wrap_ds(self.x_val, self.x_val_lable)

        val_ds = raw_ds.map(lambda x, y: (supervised_augment_eval(x, FLAGS.IMG_height, FLAGS.IMG_width, FLAGS.randaug_transform, FLAGS.randaug_magnitude),
                                          y), num_parallel_calls=AUTO)\
            .batch(self.val_batch, num_parallel_calls=AUTO)\
            .prefetch(mode_prefetch)

        logging.info("Val_ds with option")
        val_ds.with_options(options)

        return self.strategy.experimental_distribute_dataset(val_ds)
    
    @tf.function
    def Auto_Augment(self, image, policy_type, ):
        '''
        Args:
        image: A tensor [ with, height, channels]
        AutoAugment: a function to apply Policy transformation [v0, policy_simple]
        Return:
        Image: A tensor of Applied transformation [with, height, channels]
        '''

        '''
        Version 1  AutoAugmentation -->
        Custom Design still has issue Bugs in Distor function

        '''

        # augmentation_name='v1',
        '''
        if FLAGS.auto_augment == "custome":
            augmenter_apply = AutoAugment(augmentation_name='v1')

        elif FLAGS.auto_augment == "TFA_API":
            augmenter_apply = autoaug(augmentation_name='v1')
        else:
            raise ValueError("Invalid AutoAugment Implementation")
        '''
     
        augmenter_apply = autoaug(augmentation_name=policy_type)
        image = augmenter_apply.distort(image*255)

        return image / 255.

    @tf.function
    def Auto_Augment_crop(self, image, policy_type, crop_type="incpt_crp"):
        '''
        Args:
        image: A tensor [ with, height, channels]
        AutoAugment: a function to apply Policy transformation [v0, policy_simple]
        Return:
        Image: A tensor of Applied transformation [with, height, channels]
        '''

        '''
        Version 1  AutoAugmentation -->
        Custom Design still has issue Bugs in Distor function

        '''

        # augmentation_name='v1',
        '''
        if FLAGS.auto_augment == "custome":
            augmenter_apply = AutoAugment(augmentation_name='v1')

        elif FLAGS.auto_augment == "TFA_API":
            augmenter_apply = autoaug(augmentation_name='v1')
        else:
            raise ValueError("Invalid AutoAugment Implementation")
        '''
        if crop_type == "incpt_crp":
            image= simclr_augment_inception_style(image, self.IMG_SIZE)
        
        elif crop_type == "rnd_crp":
            image= simclr_augment_randcrop_global_views(image, self.IMG_SIZE)
        
        else: 
            raise ValueError("Invalid Cropping Strategies")
        
        augmenter_apply = autoaug(augmentation_name=policy_type)
        image = augmenter_apply.distort(image*255)

        return image / 255.

    @tf.function
    def Rand_Augment(self, image, num_transform=2, magnitude=7):
        '''
        Args:
        image: A tensor [ with, height, channels]
        RandAugment: a function to apply Random transformation
        Return:
        Image: A tensor of Applied transformation [with, height, channels]
        '''
        '''Version 1 RandAug Augmentation'''
        augmenter_apply = RandAugment(
            num_layers=num_transform, magnitude=magnitude)
        image = augmenter_apply.distort(image*255)

        return image[0] / 255.
    
    @tf.function
    def Rand_Augment_crop(self, image, num_transform=2, magnitude=7,crop_type="incpt_crp" ):
        '''
        Args:
        image: A tensor [ with, height, channels]
        RandAugment: a function to apply Random transformation
        Return:
        Image: A tensor of Applied transformation [with, height, channels]
        '''
        '''Version 1 RandAug Augmentation'''
        
        if crop_type == "incpt_crp":
            image= simclr_augment_inception_style(image, self.IMG_SIZE)
        
        elif crop_type == "rnd_crp":
            image= simclr_augment_randcrop_global_views(image, self.IMG_SIZE)
        
        else: 
            raise ValueError("Invalid Cropping Strategies")

        #augmenter_apply = RandAugment(
        #    num_layers=num_transform, magnitude=magnitude)
        
        # apply the extended verison of policy space of including the SimCLR
        augmenter_apply = Extend_RandAugment(
            num_layers=num_transform, magnitude=magnitude)
        image = augmenter_apply.distort(image*255)

        return image[0] / 255.

    @tf.function
    def Rand_Augment_modif(self, image, num_transform, magnitude):
        '''
        Args:
        image: A tensor [ with, height, channels]
        RandAugment: a function to apply Random transformation
        Return:
        Image: A tensor of Applied transformation [with, height, channels]
        '''
        '''Version 1 RandAug Augmentation'''
        # print(image.shape)
        augmenter_apply = Proposed_RandAugment(
            num_layers=num_transform, magnitude=magnitude)
        image = augmenter_apply.distort(image*255)

        return image [0] / 255.

    @tf.function
    def Fast_Augment(self, image, policy_type="imagenet"):
        augmenter_apply = Fast_AutoAugment(policy_type=policy_type)
        # this return (trfs_img, apply_policies)
        image = tf.py_function(augmenter_apply.distort, [
                               image*255.], Tout=[tf.float32])
        return image

    @tf.function
    def Fast_Augment_crop(self, image, policy_type="imagenet", crop_type="incpt_crp"):
        

        if crop_type == "incpt_crp":
            image= simclr_augment_inception_style(image, self.IMG_SIZE)
        
        elif crop_type == "rnd_crp":
            image= simclr_augment_randcrop_global_views(image, self.IMG_SIZE)
        
        else: 
            raise ValueError("Invalid Cropping Strategies")
            
        augmenter_apply = Fast_AutoAugment(policy_type=policy_type)
        # this return (trfs_img, apply_policies)
        image = tf.py_function(augmenter_apply.distort, [
                               image*255.], Tout=[tf.float32])
        return image

    @tf.function
    def SimCLR_Augment_crop(self, image, crop_type="incpt_crp"): 
        
        if crop_type == "incpt_crp":
            image= simclr_augment_inception_style(image, self.IMG_SIZE)
        
        elif crop_type == "rnd_crp":
            image= simclr_augment_randcrop_global_views(image, self.IMG_SIZE)
        
        else: 
            raise ValueError("Invalid Cropping Strategies")
        
        image=simclr_augment_style(image)
        
        return image

    def simclr_crop_da(self, crop_type="incpt_crp"):
        
        if not crop_type in Imagenet_dataset.crop_dict.keys():
            raise ValueError(
                f"The given cropping strategy {crop_type} is not supported")

        ds_one = self.wrap_ds(self.x_train, self.x_train_lable)
        ds_one = ds_one.shuffle(self.BATCH_SIZE * 100, seed=self.seed)

        train_ds_one = self.wrap_da(ds_one, self.crop_dict[crop_type])

        ds_two = self.wrap_ds(self.x_train, self.x_train_lable)
        ds_two = ds_one.shuffle(self.BATCH_SIZE * 100, seed=self.seed)
        train_ds_two = self.wrap_da(ds_two, self.crop_dict[crop_type])

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
        # train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        return self.strategy.experimental_distribute_dataset(train_ds)

    def AutoAug_strategy(self, crop_type="incpt_crp", policy_type="v1"):
        if not crop_type in Imagenet_dataset.crop_dict.keys():
            raise ValueError(
                f"The given cropping strategy {crop_type} is not supported")

        ds = self.wrap_ds(self.x_train, self.x_train_lable)
        ds = ds.shuffle(self.BATCH_SIZE * 100, seed=self.seed)\
        
        # unstable-version: memory issue disable
        # ds = self.get_imgnet_ds()
        

        train_ds= ds.map(lambda x, y: ((self.Auto_Augment_crop(x,policy_type=policy_type,crop_type=crop_type), y),
                            (self.Auto_Augment_crop(x,policy_type=policy_type,crop_type=crop_type), y),), num_parallel_calls=AUTO)\
                            .batch(self.BATCH_SIZE, ).prefetch(mode_prefetch)
        
        if FLAGS.dataloader: 
            print("Implement dataloader with_option")
            train_ds.with_options(options)
        else: 
            print("Not implement with option")
    
        return self.strategy.experimental_distribute_dataset(train_ds)

    # test to apply the extend version of policy space by change the source of self.Rand_Augment_crop
    def RandAug_strategy(self, crop_type="incpt_crp", num_transform=3, magnitude=4):
        if not crop_type in Imagenet_dataset.crop_dict.keys():
            raise ValueError(
                f"The given cropping strategy {crop_type} is not supported")

        ds = self.wrap_ds(self.x_train, self.x_train_lable)
        ds = ds.shuffle(self.BATCH_SIZE * 100, seed=self.seed)

        # unstable-version: memory issue disable
        # ds = self.get_imgnet_ds()
        # tf.print(f"\n\ndataset content : {ds} \n; len : {len(ds)}\n\n")
        train_ds= ds.map(lambda x, y: ((self.Rand_Augment_crop(x, num_transform, magnitude, crop_type=crop_type), y),
                                    (self.Rand_Augment_crop(x, num_transform, magnitude, crop_type=crop_type), y)),  num_parallel_calls=AUTO)\
                    .batch(self.BATCH_SIZE, num_parallel_calls=AUTO).prefetch(mode_prefetch)
       
        if FLAGS.dataloader: 
            print("Implement dataloader with_option")
            train_ds.with_options(options)
        else: 
            print("Not implement with option")
            
        return train_ds #self.strategy.experimental_distribute_dataset(train_ds)

    def FastAug_strategy(self, crop_type="incpt_crp", policy_type="imagenet"):
        if not crop_type in Imagenet_dataset.crop_dict.keys():
            raise ValueError(
                f"The given cropping strategy {crop_type} is not supported")

        ds = self.wrap_ds(self.x_train, self.x_train_lable)
        ds = ds.shuffle(self.BATCH_SIZE * 100, seed=self.seed)

        # unstable-version: memory issue disable
        #ds = self.get_imgnet_ds() 

        train_ds = ds.map(lambda x, y: ((*self.Fast_Augment_crop(x, policy_type=policy_type, crop_type=crop_type), y),
                    (*self.Fast_Augment_crop(x, policy_type=policy_type, crop_type=crop_type), y)), num_parallel_calls=AUTO)\
                .batch(self.BATCH_SIZE, num_parallel_calls=AUTO).prefetch(mode_prefetch)

        if FLAGS.dataloader: 
            print("Implement dataloader with_option")
            train_ds.with_options(options)
        else: 
            print("Not implement with option")
        
        return self.strategy.experimental_distribute_dataset(train_ds)

    def mixed_strategy(self,crop_type="incpt_crp", auto_policy_type="v1", Fast_policy_type="imagenet", num_transform=1, magnitude=10,): 
        
        if not crop_type in Imagenet_dataset.crop_dict.keys():
            raise ValueError(
                f"The given cropping strategy {crop_type} is not supported")

        ds = self.wrap_ds(self.x_train, self.x_train_lable)
        ds = ds.shuffle(self.BATCH_SIZE * 100, seed=self.seed)

        # unstable-version:
        #ds_dict = self.get_imgnet_ds()
        # tra_ds, val_ds, tst_ds = ds_dict['train'], ds_dict['validation'], ds_dict['test']

 

        train_ds = ds.map(lambda x, y: ((self.SimCLR_Augment_crop(x,crop_type=crop_type ), y),
                                        (self.Auto_Augment_crop(x,policy_type=auto_policy_type,crop_type=crop_type), y),
                                        (*self.Fast_Augment_crop(x, policy_type=Fast_policy_type, crop_type=crop_type), y),
                                        (self.Rand_Augment_crop(x, num_transform, magnitude, crop_type=crop_type), y) ), num_parallel_calls=AUTO) \
                             .batch(self.BATCH_SIZE, num_parallel_calls=AUTO).prefetch(mode_prefetch)
            
        if FLAGS.dataloader: 
            print("Implement dataloader with_option")
            train_ds.with_options(options)
        else: 
            print("Not implement with option")
            
        return self.strategy.experimental_distribute_dataset(train_ds)
     
    def Auto_and_simclr_strategy(self,crop_type="incpt_crp", auto_policy_type="v1" ): 
        
        if not crop_type in Imagenet_dataset.crop_dict.keys():
            raise ValueError(
                f"The given cropping strategy {crop_type} is not supported")

        ds = self.wrap_ds(self.x_train, self.x_train_lable)
        ds = ds.shuffle(self.BATCH_SIZE * 100, seed=self.seed)

        # unstable-version:
        #ds_dict = self.get_imgnet_ds()
        # tra_ds, val_ds, tst_ds = ds_dict['train'], ds_dict['validation'], ds_dict['test']
        train_ds = ds.map(lambda x, y: ((self.SimCLR_Augment_crop(x,crop_type=crop_type ), y),
                                        (self.SimCLR_Augment_crop(x,crop_type=crop_type ), y),
                                        (self.Auto_Augment_crop(x,policy_type=auto_policy_type,crop_type=crop_type), y),
                                        (self.Auto_Augment_crop(x,policy_type=auto_policy_type,crop_type=crop_type), y) ), num_parallel_calls=AUTO) \
                             .batch(self.BATCH_SIZE, num_parallel_calls=AUTO).prefetch(mode_prefetch)
            
        if FLAGS.dataloader: 
            print("Implement dataloader with_option")
            train_ds.with_options(options)
        else: 
            print("Not implement with option")
            
        return self.strategy.experimental_distribute_dataset(train_ds)
    
    def simclr_strategy(self, crop_type="incpt_crp",):
        if not crop_type in Imagenet_dataset.crop_dict.keys():
            raise ValueError(
                f"The given cropping strategy {crop_type} is not supported")

        ds = self.wrap_ds(self.x_train, self.x_train_lable)
        ds = ds.shuffle(self.BATCH_SIZE * 100, seed=self.seed)

        # unstable-version: memory issue disable
        # ds = self.get_imgnet_ds()
        # tf.print(f"\n\ndataset content : {ds} \n; len : {len(ds)}\n\n")
        train_ds= ds.map(lambda x, y: ((self.SimCLR_Augment_crop(x, crop_type=crop_type), y),
                                    (self.SimCLR_Augment_crop(x,  crop_type=crop_type), y)),  num_parallel_calls=AUTO)\
                    .batch(self.BATCH_SIZE, num_parallel_calls=AUTO).prefetch(mode_prefetch)
       
        if FLAGS.dataloader: 
            print("Implement dataloader with_option")
            train_ds.with_options(options)
        else: 
            print("Not implement with option")
            
        return train_ds #self.strategy.experimental_distribute_dataset(train_ds)

    # in some degree, multi-view is complete ~ ~
    def multi_view_data_aug(self, da_func=None, da_type=None):
        mv = Multi_viewer(da_inst=da_func)

        raw_ds = self.wrap_ds(self.x_train, self.x_train_lable)
        #train_ds = raw_ds.map( lambda x, y : tf.py_function(mv.multi_view, [x, y], Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]) )
        train_ds = raw_ds.map( lambda x, y : mv.multi_view (x,  y), num_parallel_calls=AUTO)
        
        #tra_ds_lst = self.wrap_da(raw_ds,  mv.multi_view, "mv_aug")
        #train_ds = tf.data.Dataset.zip(tra_ds_lst)
        logging.info("Train_ds_multiview dataloader with option")
        train_ds.with_options(options)
        return self.strategy.experimental_distribute_dataset(train_ds)

    @tf.function
    def random_resize_crop(self, image, min_scale, max_scale, crop_size):
        
        # conditional resizing
        if crop_size == 224:
            image_shape = 260
            image = tf.image.resize(image, (image_shape, image_shape))
        else:
            image_shape = 160
            image = tf.image.resize(image, (image_shape, image_shape))
        # get the crop size for given min and max scale
        size = tf.random.uniform(shape=(1,), minval=min_scale*image_shape, maxval=max_scale*image_shape, dtype=tf.float32)
        size = tf.cast(size, tf.int32)[0]
        # get the crop from the image
        crop = tf.image.random_crop(image, (size,size,3))
        crop_resize = tf.image.resize(crop, (crop_size, crop_size))    
        return crop_resize


    def multi_views_loader(self, min_scale, max_scale, crop_size, num_crops, num_transform=1, magnitude=10, policy_type="input_policy", augment_strategy="RandAug"): 
        raw_ds = self.wrap_ds(self.x_train, self.x_train_lable)
        train_ds= tuple()
        
        for i, num_crop in enumerate(num_crops): 

            for _ in range(num_crop):
                # trainloader= raw_ds.map(lambda x, y: (self.random_resize_crop(x, min_scale[i], max_scale[i], crop_size[i]),y )
                # , num_parallel_calls=AUTO)
                if augment_strategy =="RandAug": 
                    print("You implement Multi-View --> RandAugment")
                    trainloader= raw_ds.map(lambda x, y: (self.random_resize_crop(x, min_scale[i], max_scale[i], crop_size[i]),y )
                    , num_parallel_calls=AUTO)\
                    .map(lambda x, y: (self.Rand_Augment(x, num_transform, magnitude), y), num_parallel_calls=AUTO)
                elif augment_strategy =="AutoAug":
                    print("You implement Multi-View --> AutoAug") 
                    trainloader= raw_ds.map(lambda x, y: (self.random_resize_crop(x, min_scale[i], max_scale[i], crop_size[i]),y )
                    , num_parallel_calls=AUTO)\
                    .map(lambda x, y: (self.Auto_Augment(x, policy_type=policy_type), y), num_parallel_calls=AUTO)
                
                elif augment_strategy =="SimCLR":
                    print("You implement Multi-View --> SimCLR")
                    img_shp = (self.IMG_SIZE, self.IMG_SIZE)
                    trainloader = raw_ds.map(lambda x, y: (self.random_resize_crop(x, min_scale[i], max_scale[i], crop_size[i]),y ) 
                    , num_parallel_calls=AUTO).map(lambda x, y: (simclr_augment_style(x, ), y), num_parallel_calls=AUTO)

                elif augment_strategy =="FastAA":
                    print("You implement Multi-View --> FASTAA")
                    trainloader= raw_ds.map(lambda x, y: (self.random_resize_crop(x, min_scale[i], max_scale[i], crop_size[i]),y )
                    , num_parallel_calls=AUTO).map(lambda x, y: (*self.Fast_Augment(x, policy_type=policy_type), y), num_parallel_calls=AUTO)
                else: 
                    raise ValueError ("Invalid Data Augmentation Strategies")
                ## Directly apply with_option
                trainloader=trainloader.shuffle(self.BATCH_SIZE*10, seed=self.seed)
                trainloader = trainloader.with_options(options)
                train_ds +=(trainloader,)

        # Train_ds ziping multiple (train_ds_global, train_ds_local) 
        train_ds = tf.data.Dataset.zip(train_ds)

        #train_ds=train_ds.shuffle(100, seed=self.seed).batch(self.BATCH_SIZE, num_parallel_calls=AUTO).prefetch(mode_prefetch)
        train_ds=train_ds.batch(self.BATCH_SIZE, num_parallel_calls=AUTO).prefetch(mode_prefetch)
        
        return self.strategy.experimental_distribute_dataset(train_ds)

    def get_data_size(self):
        return len(self.x_train), len(self.x_val)


if __name__ == '__main__':
    # first time init the ds
    load_by_tfds()
