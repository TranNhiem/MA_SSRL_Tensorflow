from Augment_Data_utils.imagenet_dataloader_under_development import Imagenet_dataset
from Augmentation_Strategies.vis_tool import imshow_imgs
import tensorflow as tf


if __name__ == "__main__":
    import pickle
    from config.config_contrast import read_cfg
    read_cfg()
    from config.absl_mock import Mock_Flag
    flag = Mock_Flag()
    FLAGS = flag.FLAGS

    # Prepare the imagenet dataset
    strategy = tf.distribute.MirroredStrategy()
    #train_global_batch = FLAGS.train_batch_size * strategy.num_replicas_in_sync
    #val_global_batch = FLAGS.val_batch_size * strategy.num_replicas_in_sync
    # side-effect of bad manner while using the abs-path
    train_global_batch = 4
    val_global_batch = 4
    FLAGS.train_label = "/code_spec/multi_augmentation_strategies_self_supervised_learning/Augment_Data_utils/image_net_1k_lable.txt"
    FLAGS.val_label = "/code_spec/multi_augmentation_strategies_self_supervised_learning/Augment_Data_utils/ILSVRC2012_validation_ground_truth.txt"
    FLAGS.train_path = "/code_spec/ssl_ds/train"
    FLAGS.val_path =  "/code_spec/ssl_ds/val"
    train_dataset = Imagenet_dataset(img_size=FLAGS.image_size, train_batch=train_global_batch,  val_batch=val_global_batch,
                                        strategy=strategy, train_path=FLAGS.train_path, val_path=FLAGS.val_path,
                                        train_label=FLAGS.train_label, val_label=FLAGS.val_label, subset_class_num=FLAGS.num_classes )
    ## Readout the train/valid data
    #  baseline ds
    train_ds = train_dataset.simclr_crop_da("incpt_style")
    #  auto_augment ds
    auto_ds = train_dataset.auto_data_aug(da_type="auto_aug")  # customized DA strategy
    # multi-view with random augment ds
    #mv_ds = train_dataset.multi_view_data_aug(da_type="rand_aug")
    # validation ds with label 
    val_ds = train_dataset.supervised_validation()
    
    for (im_bt_1, _), (_, im_bt_2) in zip(train_ds, val_ds):
        # very ugly object
        with open('./ds.pickle', 'wb') as f:
            pickle.dump(im_bt_1, f)
        break

    for (im_bt_1, _) in auto_ds:
        # very ugly object
        with open('./auto_ds.pickle', 'wb') as f:
            pickle.dump(im_bt_1, f)
        tmp = im_bt_1[0].values
        #tmp = tmp[0].numpy()
        print(len(tmp))
        break

    