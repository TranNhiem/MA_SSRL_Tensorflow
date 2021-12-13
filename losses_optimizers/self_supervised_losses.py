import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

######################################################################################
'''Supervised  Contrastive LOSS'''
######################################################################################


def multiclass_npair_loss(z, y):
    '''
    arg: z, hidden feature vectors(B_S[z], n_features)
    y: ground truth of shape (B_S[z])

    '''
    # Cosine similarity matrix
    z = tf.math.l2_normalize(z,  axis=1)
    Similarity = tf.matmul(z, z, transpose_b=True)
    loss = tfa.losses.npairs_loss(y, Similarity)
    return loss

# Supervised Contrastive Learning Paper


def multi_class_npair_loss_temperature(z, y, temperature):
    x_feature = tf.math.l2_normalize(z,  axis=1)
    similarity = tf.divide(
        tf.matmul(x_feature, tf.transpose(x_feature)), temperature)
    return tfa.losses.npairs_loss(y, similarity)


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



######################################################################################
'''NON-CONTRASTIVE LOSS'''
####################################################################################

'''BYOL SYMETRIZE LOSS'''
# Symetric LOSS


def byol_symetrize_loss(p, z, temperature):
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

def byol_symetrize_mixed_loss(p, z, lamda, temperature ): 
    '''
    Arg: 
        p, z : Augmented Feature from img_1, img_2 
        lamda: mix percentage value 
        temperature: scaling term for the loss function 
    Return:  the mixed loss 

    '''
    p = tf.math.l2_normalize(p, axis=1)  # (2*bs, 128)
    z = tf.math.l2_normalize(z, axis=1)  # (2*bs, 128)
    # Calculate contrastive Loss
    batch_size = tf.shape(p)[0]
    similarities=[]
    for i in range(len(lamda)): 
        # Measure similarity
        similarity = lamda[i]*(tf.multiply(p[i], z[i])))
        similarities.append(similarity)
        loss = 2 - 2 * tf.reduce_mean(similarities)
    # 
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    logits_ab = tf.matmul(p, z, transpose_b=True) / temperature
 


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
