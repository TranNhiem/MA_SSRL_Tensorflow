import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


######################################################################################
'''Self-Supervised CONTRASTIVE LOSS'''
######################################################################################

'''N-Pair Loss'''


def multiclass_N_pair_loss(p, z):
    x_i = tf.math.l2_normalize(p, axis=1)
    x_j = tf.math.l2_normalize(z, axis=1)
    similarity = tf.matmul(x_i, x_j, transpose_b=True)
    batch_size = tf.shape(p)[0]
    contrastive_labels = tf.range(batch_size)

    # Simlarilarity treat as logic input for Cross Entropy Loss
    # Why we need the Symmetrized version Here??
    loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
        contrastive_labels, similarity, from_logits=True)
    loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
        contrastive_labels, tf.transpose(similarity), from_logits=True)

    return (loss_1_2+loss_2_1)/2


'''SimCLR Paper Nt-Xent Loss Keras Version'''
# Nt-Xent Loss Symmetrized


def nt_xent_symmetrize_keras(p, z, temperature):
    # cosine similarity the dot product of p,z two feature vectors
    x_i = tf.math.l2_normalize(p, axis=1)
    x_j = tf.math.l2_normalize(z, axis=1)
    similarity = (tf.matmul(x_i, x_j, transpose_b=True)/temperature)
    # the similarity from the same pair should be higher than other views
    batch_size = tf.shape(p)[0]  # Number Image within batch
    contrastive_labels = tf.range(batch_size)

    # Simlarilarity treat as logic input for Cross Entropy Loss
    # Why we need the Symmetrized version Here??
    loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
        contrastive_labels, similarity, from_logits=True,)  # reduction=tf.keras.losses.Reduction.SUM
    loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
        contrastive_labels, tf.transpose(similarity), from_logits=True, )
    return (loss_1_2 + loss_2_1) / 2


'''SimCLR paper Asytemrize_loss V2'''

# Mask to remove the positive example from the rest of Negative Example


def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images
    # Ensure distinct pair of image get their similarity scores
    # passed as negative examples
    batch_size = batch_size.numpy()
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i+batch_size] = 0
    return tf.constant(negative_mask)


consie_sim_1d = tf.keras.losses.CosineSimilarity(
    axis=1, reduction=tf.keras.losses.Reduction.NONE)
cosine_sim_2d = tf.keras.losses.CosineSimilarity(
    axis=2, reduction=tf.keras.losses.Reduction.NONE)


def nt_xent_asymetrize_loss_v1(p, z, temperature):  # negative_mask

    # L2 Norm
    batch_size = tf.shape(p)[0]
    batch_size = tf.cast(batch_size, tf.int32)
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    p_l2 = tf.math.l2_normalize(p, axis=1)
    z_l2 = tf.math.l2_normalize(z, axis=1)

    # Cosine Similarity distance loss

    # pos_loss = consie_sim_1d(p_l2, z_l2)
    pos_loss = tf.matmul(tf.expand_dims(p_l2, 1), tf.expand_dims(z_l2, 2))

    pos_loss = (tf.reshape(pos_loss, (batch_size, 1)))/temperature

    negatives = tf.concat([p_l2, z_l2], axis=0)
    # Mask out the positve mask from batch of Negative sample
    negative_mask = get_negative_mask(batch_size)

    loss = 0
    for positives in [p_l2, z_l2]:

        # negative_loss = cosine_sim_2d(positives, negatives)
        negative_loss = tf.tensordot(tf.expand_dims(
            positives, 1), tf.expand_dims(tf.transpose(negatives), 0), axes=2)
        l_labels = tf.zeros(batch_size, dtype=tf.int32)
        l_neg = tf.boolean_mask(negative_loss, negative_mask)

        l_neg = tf.reshape(l_neg, (batch_size, -1))
        l_neg /= temperature

        logits = tf.concat([pos_loss, l_neg], axis=1)  # [N, K+1]

        loss_ = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        loss += loss_(y_pred=logits, y_true=l_labels)

    batch_size = tf.cast(batch_size, tf.float32)
    loss = loss/(2*batch_size)
    return loss


'''SimCLR Paper Nt-Xent Loss # SYMETRIC Loss'''
# Nt-Xent ---> N_Pair loss with Temperature scale
# Nt-Xent Loss (Remember in this case Concatenate Two Tensor Together)


def nt_xent_asymetrize_loss_v2(z,  temperature):
    '''The issue of design this loss two image is in one array
    when we multiply them that will lead two two same things mul together???

    '''
    # Feeding data (ALready stack two version Augmented Image)[2*bs, 128]
    z = tf.math.l2_normalize(z, axis=1)
    similarity_matrix = tf.matmul(
        z, z, transpose_b=True)  # pairwise similarity

    similarity = tf.exp(similarity_matrix / temperature)

    logit_output = similarity_matrix / temperature
    batch_size = tf.shape(z)[0]

    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)

    ij_indices = tf.reshape(tf.range(z.shape[0]), shape=[-1, 2])
    ji_indices = tf.reverse(ij_indices, axis=[1])

    #[[0, 1], [1, 0], [2, 3], [3, 2], ...]
    positive_indices = tf.reshape(tf.concat(
        [ij_indices, ji_indices], axis=1), shape=[-1, 2])  # Indice positive pair

    # --> Output N-D array
    numerator = tf.gather_nd(similarity, positive_indices)
    # 2N-1 (sample)
    # mask that discards self-similarity
    negative_mask = 1 - tf.eye(z.shape[0])

    # compute sume across dimensions of Tensor (Axis is important in this case)
    # None sum all element scalar, 0 sum all the row, 1 sum all column -->1D metric
    denominators = tf.reduce_sum(
        tf.multiply(negative_mask, similarity), axis=1)
    losses = -tf.math.log(numerator/denominators)
    total_loss = tf.reduce_mean(losses)

    return total_loss, logit_output, labels


def nt_xent_symetrize_loss_simcrl(hidden1, hidden2, LARGE_NUM,
                                  hidden_norm=True,
                                  temperature=1.0,
                                  ):
    """Compute loss for model.

    Args:
      hidden: hidden vector (`Tensor`) of shape (bsz, dim).
      hidden_norm: whether or not to use normalization on the hidden vector.
      temperature: a `floating` number for temperature scaling.

    Returns:
      A loss scalar.
      The logits for contrastive prediction task.
      The labels for contrastive prediction task.
    """
    # Get (normalized) hidden1 and hidden2.
    if hidden_norm:
        hidden1 = tf.math.l2_normalize(hidden1, -1)  # 1
        hidden2 = tf.math.l2_normalize(hidden2, -1)
    #hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]

    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1_large,
                          transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2_large,
                          transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM

    logits_ab = tf.matmul(hidden1, hidden2_large,
                          transpose_b=True) / temperature

    logits_ba = tf.matmul(hidden2, hidden1_large,
                          transpose_b=True) / temperature

    loss_a = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ba, logits_bb], 1))
    loss = tf.reduce_mean(loss_a + loss_b) / 2

    return loss, logits_ab, labels


######################################################################################
'''NON-CONTRASTIVE LOSS'''
####################################################################################


def byol_loss(p, z, temperature):
    p = tf.math.l2_normalize(p, axis=1)  # (2*bs, 128)
    z = tf.math.l2_normalize(z, axis=1)  # (2*bs, 128)
    # Calculate contrastive Loss
    batch_size = tf.shape(p)[0]
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    logits_ab = tf.matmul(p, z, transpose_b=True) / temperature
    # Measure similarity
    similarities = tf.reduce_sum(tf.multiply(p, z), axis=1)
    loss = 2 - 2 * tf.reduce_mean(similarities)
    return loss, logits_ab, labels

def byol_loss_v1(p, z, temperature):
    p = tf.math.l2_normalize(p, axis=1)  # (2*bs, 128)
    z = tf.math.l2_normalize(z, axis=1)  # (2*bs, 128)
    # Calculate contrastive Loss
    batch_size = tf.shape(p)[0]
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    logits_ab = tf.matmul(p, z, transpose_b=True) / temperature
    # Measure similarity
    similarities = tf.reduce_sum(tf.multiply(p, z), axis=1)
    #loss = 2 - 2 * similarities
    return similarities, logits_ab, labels


def byol_multi_views_loss(v1, v2, v3, v4, v5, temperature, alpha):
    

    glob_loss, logits_ab, labels= byol_loss(v1, v2, temperature)
    
    loc_loss_1, _, _ = byol_loss(v3, v4, temperature)
    loc_loss_2, _, _= byol_loss(v3, v5, temperature)
    local_loss = loc_loss_1 + loc_loss_2
    
    loss= alpha*glob_loss +  (1-alpha)* local_loss

    return loss, logits_ab, labels

def byol_2_augmentation_loss(v1, v2, v3, v4,  temperature, weight_loss=0.5):

    loss_aug1, logits_ab, labels= byol_loss_v1(v1, v2, temperature)
    loss_aug2, _, _= byol_loss_v1(v3, v4, temperature)

    loss= weight_loss*loss_aug1 +  (1.0-weight_loss)* loss_aug2

    return loss, logits_ab, labels

def byol_mixed_loss(p, z, p_z_mix, lamda, alpha, temperature):
    '''
    Arg: 
        p, z : Augmented Feature from img_1, img_2 
        lamda: mix percentage value 
        alpha: is weigted loss control
        temperature: "Just for 
    Return:  the mixed loss 

    '''
    batch_size = tf.shape(p)[0]
    one_tensor = tf.ones(shape=(batch_size, 1), dtype=tf.dtypes.float32, )
    # Image similarity
    image_loss, logit, lable = byol_symetrize_loss(p, z, temperature)

    # Normal augmented image 1 vs mixed image
    normal_mix_loss, _ = byol_symetrize_loss(p, p_z_mix, temperature)

    # Reverse Order Image 2 vs mixed image
    reverse_mix_loss, _ = byol_symetrize_loss(p, p_z_mix, temperature)

    loss = (image_loss + (lamda*normal_mix_loss +
            (one_tensor-lamda)*reverse_mix_loss))/2

    return loss, logit, lable


'''Loss 2 SimSiam Model'''
# Asymetric LOSS


def simsam_loss(p, z):
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))


def simsam_loss_non_stop_Gr(p, z):
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    #z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))
