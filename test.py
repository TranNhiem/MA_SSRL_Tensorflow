from Augment_Data_utils.imagenet_dataloader_under_development import Imagenet_dataset
import tensorflow as tf

if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    train_dataset = Imagenet_dataset(img_size=2, train_batch=1,  val_batch=1,
                                        strategy=strategy, train_path=1, val_path=1,
                                        train_label=1, val_label=1, subset_class_num=1 )