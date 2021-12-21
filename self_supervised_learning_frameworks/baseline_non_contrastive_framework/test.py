# for disable tf warning message..
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from Augment_Data_utils.imagenet_dataloader_under_development import Imagenet_dataset
from Augmentation_Strategies.vis_tool import imshow_imgs


def dump_pk(file_name='./test.pickle', dump_obj=None):
    with open(file_name, 'wb') as pk_ptr:
        pickle.dump(dump_obj, pk_ptr)

def prnt_2view(im_bt, lab_bt):  # with 8 gpus [8 elem] -> (4 batch_size), (4 empty)
    print(f"size of view(8 gpus) : {len(im_bt)}, img shape : {im_bt[0].shape}\n")
    #print(f"non-thing shape : {im_bt[4].shape}\n")
    #print(f"corresponding label (8) : {len(lab_bt)}, cls type (200 class): {lab_bt[0].shape}\n")
    #print(f"non-thing shape : {lab_bt[4].shape}\n")

def test_traVal(train_dataset):
    train_ds = train_dataset.simclr_crop_da("incpt_style")
    val_ds = train_dataset.supervised_validation()
    
    for (im_bt_1, lab_bt_1), (im_bt_2, lab_bt_2) in train_ds:
        # un-wrap PerReplica objs
        im_bt_1, im_bt_2 = im_bt_1.values, im_bt_2.values
        lab_bt_1, lab_bt_2 = lab_bt_1.values, lab_bt_2.values

        prnt_2view(im_bt_1, lab_bt_1) ; prnt_2view(im_bt_2, lab_bt_2)
        break

    for (im_bt_1, lab_bt_1), (im_bt_2, lab_bt_2) in train_ds:
         # un-wrap PerReplica objs
        im_bt_1, im_bt_2 = im_bt_1.values, im_bt_2.values
        lab_bt_1, lab_bt_2 = lab_bt_1.values, lab_bt_2.values

        prnt_2view(im_bt_1, lab_bt_1) ; prnt_2view(im_bt_2, lab_bt_2)
        break


def test_autods(train_dataset):
    auto_ds = train_dataset.auto_data_aug(da_type="auto_aug")  # customized DA strategy
    
    for (im_bt_1, lab_bt_1), (im_bt_2, lab_bt_2) in auto_ds:
         # un-wrap PerReplica objs
        im_bt_1, im_bt_2 = im_bt_1.values, im_bt_2.values
        lab_bt_1, lab_bt_2 = lab_bt_1.values, lab_bt_2.values

        prnt_2view(im_bt_1, lab_bt_1) ; prnt_2view(im_bt_2, lab_bt_2)
        break


def test_mvds(train_dataset):
    mv_ds = train_dataset.multi_view_data_aug(da_type="rand_aug")

    for (im_bt_1, lab_bt_1), (_, _), (_, _) in auto_ds:
         # un-wrap PerReplica objs
        im_bt_1, im_bt_2 = im_bt_1.values, im_bt_2.values
        lab_bt_1, lab_bt_2 = lab_bt_1.values, lab_bt_2.values

        break

if __name__ == "__main__":
    import pickle
    import tensorflow_addons as tfa
    from config.config_contrast import read_cfg
    read_cfg()

    from config.absl_mock import Mock_Flag
    flag = Mock_Flag()
    FLAGS = flag.FLAGS
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[0:2], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)
            
    # Prepare the imagenet dataset
    strategy = tf.distribute.MirroredStrategy()
    train_global_batch = FLAGS.train_batch_size * strategy.num_replicas_in_sync
    val_global_batch = FLAGS.val_batch_size * strategy.num_replicas_in_sync
    # side-effect of bad manner while using the abs-path
    FLAGS.train_label = "/code_spec/multi_augmentation_strategies_self_supervised_learning/Augment_Data_utils/image_net_1k_lable.txt"
    FLAGS.val_label = "/code_spec/multi_augmentation_strategies_self_supervised_learning/Augment_Data_utils/ILSVRC2012_validation_ground_truth.txt"
    FLAGS.train_path = "/code_spec/ssl_ds/train"
    FLAGS.val_path =  "/code_spec/ssl_ds/val"
    train_dataset = Imagenet_dataset(img_size=FLAGS.image_size, train_batch=train_global_batch,  val_batch=val_global_batch,
                                        strategy=strategy, train_path=FLAGS.train_path, val_path=FLAGS.val_path,
                                        train_label=FLAGS.train_label, val_label=FLAGS.val_label, subset_class_num=FLAGS.num_classes )
    # Readout the train/valid data testing
    train_dataset = strategy.experimental_distribute_dataset(train_ds)
    test_traVal(train_dataset)
    
    #  Readout the from auto_augment strategies
    #test_autods(train_dataset)
    
    # multi-view with random augment ds
    #test_mvds(train_dataset)
    
    
## record the return PerReplica obj:
    #   it will return the predicted tensor calculated from each GPUs devices.
    #    for ex. we have 8 gpus on this ai01 node with setting 'batch size equ 4', 
    #            so it return 2 view (tuple) with each contain 2 sub-tuple.
    #            ,and each view contains (image, label) pair.
    #    @ un-packing image :
    #      tnsr_lst = im_bt_1[0].values, tnsr_lst : 8 view of EagerTensor
    #      tnsr_lst[0-3] : '4' image EagerTensor with shape ([1, 224, 224, 3]) 
    #      tnsr_lst[4-7] : rest 4 tf.Tensor([], shape=(0, 224, 224, 3), dtype=float32)
    #    @ un-packing label :
    #      tnsr_lst = im_bt_1[1].values, tnsr_lst : 8 view of EagerTensor
    #      tnsr_lst[0-3] : '4' label Tensor Shape([1, 200])
    #      tnsr_lst[4-7] : rest 4 tf.Tensor([], shape=(0, 200), dtype=float32)