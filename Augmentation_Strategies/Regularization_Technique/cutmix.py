'''
This is implementation Cutmix 

1. Implementation for Two images (You can implement this by youself)
2. Implementation for Random Image within mini Batch_size (Current Implement)
## The implementation in Google Colab
https://colab.research.google.com/drive/1xPHC5kRFmSi30KsTiR_sLMwcMvOcvESs?usp=sharing
'''
import tensorflow as tf
import matplotlib.pyplot as plt

SEED = 26
AUTO = tf.data.experimental.AUTOTUNE
IMG_SIZE = 32
BATCH_SIZE = 32

# sample Beta Distribution


class Dataset_mixture():

    def __init__(self, IMG_SIZE, BATCH_SIZE):

        self.IMG_SIZE = IMG_SIZE
        self.BATCH_SIZE = BATCH_SIZE

        (self.x_train, self.y_train), (self.x_test,
                                       self.y_test) = tf.keras.datasets.cifar10.load_data()

        # Normalize training and testing images
        self.x_train = tf.cast(self.x_train, tf.float32)  # /255.
        self.x_test = tf.cast(self.x_test, tf.float32)

        self.y_train = tf.cast(tf.squeeze(self.y_train), tf.float32)
        self.y_test = tf.cast(tf.squeeze(self.y_test), tf.float32)
        self.y_train = tf.keras.utils.to_categorical(
            self.y_train, num_classes=10)

    @classmethod
    def sample_beta_distribution_cutmix(self, size, concentration_0=0.4, concentration_1=0.4):
        gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    @classmethod
    @tf.function
    def get_box(self, IMG_SIZE, lambda_value):
        cut_rat = tf.math.sqrt(1.0 - lambda_value)

        cut_w = int(IMG_SIZE) * cut_rat  # rws
        cut_w = tf.cast(cut_w, tf.int32)

        cut_h = int(IMG_SIZE) * cut_rat  # rh
        cut_h = tf.cast(cut_h, tf.int32)

        cut_x = tf.random.uniform(
            (1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)  # rx
        cut_y = tf.random.uniform(
            (1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)  # ry

        boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, IMG_SIZE)
        boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, IMG_SIZE)
        bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, IMG_SIZE)
        bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, IMG_SIZE)

        target_h = bby2 - boundaryy1
        if target_h == 0:
            target_h += 1

        target_w = bbx2 - boundaryx1
        if target_w == 0:
            target_w += 1

        #print(f'Size of BBox, and Target_size checking{boundaryx1, boundaryy1,target_h, target_w }')

        return boundaryx1, boundaryy1, target_h, target_w

    @classmethod
    @tf.function
    def cutmix(self, train_ds_one, train_ds_two, IMG_SIZE, alpha_cutmix):
        (image1, label1), (image2, label2) = train_ds_one, train_ds_two

        # alpha_cutmix = [0.3]
        # alpha_cutmix = [0.3]

        # Get a sample from the Beta distribution
        lambda_value = self.sample_beta_distribution_cutmix(
            1, alpha_cutmix, alpha_cutmix)

        # Define Lambda
        lambda_value = lambda_value[0][0]

        # Get the bounding box offsets, heights and widths
        boundaryx1, boundaryy1, target_h, target_w = self.get_box(
            IMG_SIZE, lambda_value)

        # Get a patch from the second image (`image2`)
        crop2 = tf.image.crop_to_bounding_box(
            image2, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image2` patch (`crop2`) with the same offset
        image2 = tf.image.pad_to_bounding_box(
            crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE
        )
        # Get a patch from the first image (`image1`)
        crop1 = tf.image.crop_to_bounding_box(
            image1, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image1` patch (`crop1`) with the same offset
        img1 = tf.image.pad_to_bounding_box(
            crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE
        )

        # Modify the first image by subtracting the patch from `image1`
        # (before applying the `image2` patch)
        image1 = image1 - img1
        # Add the modified `image1` and `image2`  together to get the CutMix image
        image = image1 + image2

        # Adjust Lambda in accordance to the pixel ration
        lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
        lambda_value = tf.cast(lambda_value, tf.float32)

        # Combine the labels of both images
        label = lambda_value * label1 + (1 - lambda_value) * label2

        return image, label

    def cutmix_dataset(self, alpha_cutmix):

        train_ds_one = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                        .shuffle(self.BATCH_SIZE * 100)
                        # .batch(BATCH_SIZE)
                        .map(lambda x, y: (x/255., y),
                             num_parallel_calls=AUTO,)
                        .prefetch(AUTO)
                        )

        train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                        .shuffle(self.BATCH_SIZE * 100)
                        .map(lambda x, y: (x/255., y),
                             num_parallel_calls=AUTO,)
                        # .batch(BATCH_SIZE)
                        .prefetch(AUTO)
                        )
        # Combine two shuffled datasets from the same training data.
        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        train_ds_cmu = (
            train_ds.shuffle(1024)
            .map(lambda ds_one, ds_two: self.cutmix(ds_one, ds_two, self.IMG_SIZE, alpha_cutmix), num_parallel_calls=AUTO)
            .batch(self.BATCH_SIZE)
            .prefetch(AUTO)
        )

        return train_ds_cmu


# Visualization CutMix Result
Cutmix = Dataset_mixture(IMG_SIZE, BATCH_SIZE)

cutmix_alpha = [0.4]
Cutmix_ds = Cutmix.cutmix_dataset(cutmix_alpha)
# Ploting and Visualization Result
sample_images, sample_labels = next(iter(Cutmix_ds))

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(zip(sample_images[:9], sample_labels[:9])):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().squeeze())
    print(label.numpy().tolist())
    plt.axis("off")
