# for disable tf warning message..
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pickle
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from Augment_Data_utils.imagenet_dataloader_under_development import Imagenet_dataset
from Augmentation_Strategies.vis_tool import imshow_imgs

def set_gpu_env(n_gpu=8):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[0:n_gpu], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)

def dump_pk(file_name='./test.pickle', dump_obj=None):
    with open(file_name, 'wb') as pk_ptr:
        pickle.dump(dump_obj, pk_ptr)

def prnt_2view(im_bt, lab_bt, batch_size):
    # im_bt : EagerTensor, .numpy() API convert into numpy array
    print(f"origin setting batch_size : {batch_size}\n")
    print(f"img batch shape: {im_bt.shape}, lab batch shape : {lab_bt.shape}\n")
    print(f"valid img range --> max : {im_bt.numpy().max()}, min : {im_bt.numpy().min()}\n")
    
def test_traVal(train_dataset):
    train_ds = train_dataset.simclr_crop_da("incpt_style")
    val_ds = train_dataset.supervised_validation()
    
    for (im_bt_1, lab_bt_1), (im_bt_2, lab_bt_2) in train_ds:
        print("train ds 2 view")
        prnt_2view(im_bt_1, lab_bt_1, FLAGS.train_batch_size)
        prnt_2view(im_bt_2, lab_bt_2, FLAGS.train_batch_size)
        break

    for (im_bt, lab_bt) in val_ds:
        print("valid ds 1 view")
        prnt_2view(im_bt, lab_bt, FLAGS.val_batch_size)
        break


def test_autods(train_dataset):
    auto_ds = train_dataset.auto_data_aug(da_type="rand_aug")  # customized DA strategy
    
    for (im_bt_1, lab_bt_1), (im_bt_2, lab_bt_2) in auto_ds:
        prnt_2view(im_bt_1, lab_bt_1, FLAGS.train_batch_size)
        prnt_2view(im_bt_2, lab_bt_2, FLAGS.train_batch_size)
        break


def test_mvds(train_dataset):
    mv_ds = train_dataset.multi_view_data_aug(da_type="auto_aug")
    
    for im1, im2, im3, im4, im5 in mv_ds:
        '''
        dump_pk(file_name='./im1.pickle', dump_obj=im1)
        dump_pk(file_name='./im2.pickle', dump_obj=im2)
        dump_pk(file_name='./im3.pickle', dump_obj=im3)
        dump_pk(file_name='./im4.pickle', dump_obj=im4)
        dump_pk(file_name='./im5.pickle', dump_obj=im5)
        '''
        print(f"img batch shape: {im1.shape}\n")
        print(f"valid img range --> max : {im1.numpy().max()}, min : {im1.numpy().min()}\n")
        break


if __name__ == "__main__":
    from config.config_contrast import read_cfg
    read_cfg()

    from config.absl_mock import Mock_Flag
    flag = Mock_Flag()
    FLAGS = flag.FLAGS
    
    set_gpu_env(n_gpu=8)
            
    # Prepare the imagenet dataset
    strategy = tf.distribute.MirroredStrategy()
    train_global_batch = FLAGS.train_batch_size * strategy.num_replicas_in_sync
    val_global_batch = FLAGS.val_batch_size * strategy.num_replicas_in_sync

    # side-effect of bad manner while using the abs-path
    FLAGS.train_label = "/code_spec/Augment_Data_utils/image_net_1k_lable.txt"
    FLAGS.val_label = "/code_spec/Augment_Data_utils/ILSVRC2012_validation_ground_truth.txt"
    FLAGS.train_path = "/data/train"
    FLAGS.val_path =  "/data/val"
    train_dataset = Imagenet_dataset(img_size=FLAGS.image_size, train_batch=train_global_batch,  val_batch=val_global_batch,
                                        strategy=strategy, train_path=FLAGS.train_path, val_path=FLAGS.val_path,
                                        train_label=FLAGS.train_label, val_label=FLAGS.val_label, subset_class_num=FLAGS.num_classes )
    # Readout the train/valid data testing
    #test_traVal(train_dataset)
    
    #  Readout the from auto_augment strategies
    test_autods(train_dataset)
    
    # multi-view with random augment ds
    #test_mvds(train_dataset)