'''
Demonstration Implementaiton Mixup Input 

+ 1 Strategy mix for (Two Images )
+ 2 Strategy mix for (Multiple images within Batch) 

#Real Implementation ON GoogleColab
https://colab.research.google.com/drive/1MzadOkYUiNKui8Y9YmAHlBOogukc5iCk?usp=sharing 

'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers


# Demonstration Cifar 10 Dataset
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 64
EPOCHS = 10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32")  # / 255.0
y_train = y_train.astype("float32")
y_train = tf.one_hot(y_train, 10)


train_ds_one = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(BATCH_SIZE * 100)
    # .batch(BATCH_SIZE)
)
train_ds_two = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(BATCH_SIZE * 100)
    # .batch(BATCH_SIZE)
)
# Because we will be mixing up the images and their corresponding labels, we will be
# combining two shuffled datasets from the same training data.
train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))


########################################################################
# Helper Function
########################################################################

# This function to finding the Beta Distribution for a given ALPHA Mix Value
def sample_beta_distribution(size, concentration_0=0.1, concentration_1=0.1):

    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


########################################################################
# Section 1 Mix Two images
########################################################################

# The Mixup function for Given Two Image and Alpha Values
def mix_up(ds_one, ds_two, alpha=0.2):
    '''Args
    ds_one: include image_one and label(optional)
    ds_two: include image_two and label(optional)
    '''
    # Unpack two datasets
    images_one, labels_one = ds_one
    print(images_one.shape)
    images_two, labels_two = ds_two
    batch_size = 1  # tf.shape(images_one)[0]
    print(batch_size)

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (1, 1,  1))
    y_l = tf.reshape(l, (1, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)


# First create the new dataset using our `mix_up` utility
train_ds_mu = (train_ds.map(
    lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2), num_parallel_calls=AUTO
)
    .batch(BATCH_SIZE)
)

# Ploting and Visualization Result
sample_images, sample_labels = next(iter(train_ds_mu))
sample_labels = tf.reshape(sample_labels, shape=(64, 10))

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(zip(sample_images[:9], sample_labels[:9])):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().squeeze())
    print(label.numpy().tolist())
    plt.axis("off")


########################################################################
# Section 2 Mix with Random two images within Mini Batch_size
########################################################################
train_ds_one = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(BATCH_SIZE * 100)
    .batch(BATCH_SIZE)
)
train_ds_two = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(BATCH_SIZE * 100)
    # adding Batch_image
    .batch(BATCH_SIZE)
)

train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))


def mix_up_batch(ds_one, ds_two, alpha=0.2):
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)


# First create the new dataset using our `mix_up` utility
train_ds_mu_batch = train_ds.map(
    lambda ds_one, ds_two: mix_up_batch(ds_one, ds_two, alpha=0.2), num_parallel_calls=AUTO
)

# Ploting and Visualization Result
sample_images, sample_labels = next(iter(train_ds_mu_batch))

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(zip(sample_images[:9], sample_labels[:9])):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().squeeze())
    print(label.numpy().tolist())
    plt.axis("off")
