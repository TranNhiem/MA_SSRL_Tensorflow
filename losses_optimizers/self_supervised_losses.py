import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch
import tensorflow_addons as tfa
import numpy as np
from tensorflow.keras import layers

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


'''SimCLR Paper Nt-Xent Loss # ASYMETRIC Loss'''
# Nt-Xent ---> N_Pair loss with Temperature scale
# Nt-Xent Loss (Remember in this case dataset two image stacked)
# This implementation for the Special input data


def nt_xent_asymetrize_loss(z,  temperature):
    '''The issue of design this loss two image is in one array
    when we multiply them that will lead two two same things mul together???
    '''
    # Feeding data (ALready stack two version Augmented Image)[2*bs, 128]
    z = tf.math.l2_normalize(z, axis=1)

    similarity_matrix = tf.matmul(
        z, z, transpose_b=True)  # pairwise similarity
    similarity = tf.exp(similarity_matrix / temperature)

    ij_indices = tf.reshape(tf.range(z.shape[0]), shape=[-1, 2])
    ji_indices = tf.reverse(ij_indices, axis=[1])
    # [[0, 1], [1, 0], [2, 3], [3, 2], ...]
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
    return tf.reduce_mean(losses)


'''SimCLR paper Asytemrize_loss'''

# Mask to remove the positive example from the rest of Negative Example


def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images
    # Ensure distinct pair of image get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i+batch_size] = 0

    return tf.constant(negative_mask)


# Cosine Similarity distance loss measurement
consie_sim_1d = tf.keras.losses.CosineSimilarity(
    axis=1, reduction=tf.keras.losses.Reduction.NONE)
cosine_sim_2d = tf.keras.losses.CosineSimilarity(
    axis=2, reduction=tf.keras.losses.Reduction.NONE)

# Official Implementation NCE Loss (Noisy Contrasitve Estimator Loss)


def nt_xent_asymetrize_loss_v2(p, z, temperature, batch_size):  # negative_mask
    # L2 Norm
    p_l2 = tf.math.l2_normalize(p, axis=1)
    z_l2 = tf.math.l2_normalize(z, axis=1)

    # Cosine Similarity distance loss
    # pos_loss = consie_sim_1d(p_l2, z_l2)
    pos_loss = tf.matmul(tf.expand_dims(p_l2, 1), tf.expand_dims(z_l2, 2))
    pos_loss = tf.reshape(pos_loss, (batch_size, 1))
    pos_loss /= temperature

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
        tf.keras.losses.SparseCategoricalCrossentropy()
        loss_ = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        loss += loss_(y_pred=logits, y_true=l_labels)

    loss = loss/(2*batch_size)

    return loss


'''SimCLR Paper Nt-Xent Loss # SYMMETRIZED Loss'''
# Nt-Xent Loss Symmetrized
# Simple Implemenation the loss NOTED [Experimental design]


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


######################################################################################
'''NONE CONTRASTIVE LOSS'''
####################################################################################

'''BYOL SYMETRIZE LOSS'''
# Symetric LOSS
# Offical Implementation

# Noted consider using the stop gradient here
"""already configure in Custom training loop so 
==-> you might not need the stop GRAD 
"""


def byol_symetrize_loss(p, z):

    p = tf.math.l2_normalize(p, axis=1)  # (2*bs, 128)
    z = tf.math.l2_normalize(z, axis=1)  # (2*bs, 128)

    similarities = tf.reduce_sum(tf.multiply(p, z), axis=1)
    return 2 - 2 * tf.reduce_mean(similarities)

def binary_byol_symetrize_loss(p, z, p_1, z_1): 
    
    ## L2 Norm feature Vectors
    p = tf.math.l2_normalize(p, axis=1)  # (2*bs, 128)
    z = tf.math.l2_normalize(z, axis=1)  # (2*bs, 128)
    p_1 = tf.math.l2_normalize(p_1, axis=1)  # (2*bs, 128)
    z_2 = tf.math.l2_normalize(z_1, axis=1)  # (2*bs, 128)

    ## Object + Background Loss
    similarities_object = tf.reduce_sum(tf.multiply(p, z), axis=1)
    similarities_backgroud = tf.reduce_sum(tf.multiply(p_1, z_1), axis=1)
    loss_1= 2 - 2 * tf.reduce_mean(similarities_object)
    loss_2=2 - 2 * tf.reduce_mean(similarities_backgroud)
    total_loss = loss_1 + loss_2
    
    return total_loss


def semantic_byol_symetrize_loss(p, z): 
    '''
    Args: 
        P: array of Semantic features
        Z: array of sematinc features 
    '''
    total_loss=0
    ## Get the total number of semantic features
    for i in range(len(p)): 
        p_norm =tf.math.l2_normalize(p[i], axis=1)
        z_norm =tf.math.l2_normalize(z[i], axis=1)

        similarities_object = tf.reduce_sum(tf.multiply(p_norm, z_norm), axis=1)
        loss= 2 - 2 * tf.reduce_mean(similarities_object)
        total_loss +=loss

    return total_loss




def binary_byol_symetrize_loss(p, z, p_1, z_1): 
    
    ## L2 Norm feature Vectors
    p = tf.math.l2_normalize(p, axis=1)  # (2*bs, 128)
    z = tf.math.l2_normalize(z, axis=1)  # (2*bs, 128)
    p_1 = tf.math.l2_normalize(p_1, axis=1)  # (2*bs, 128)
    z_2 = tf.math.l2_normalize(z_1, axis=1)  # (2*bs, 128)

    ## Object + Background Loss
    similarities_object = tf.reduce_sum(tf.multiply(p, z), axis=1)
    similarities_backgroud = tf.reduce_sum(tf.multiply(p_1, z_1), axis=1)
    loss_1= 2 - 2 * tf.reduce_mean(similarities_object)
    loss_2=2 - 2 * tf.reduce_mean(similarities_backgroud)
    total_loss = loss_1 + loss_2
    
    return total_loss




'''Loss 2 SimSiam Model'''
# Asymetric LOSS
# offical Implementation
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
# Experimental testing Collapse Situation


def simsam_loss_non_stop_Gr(p, z):
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    # z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))


######################################################################################
'''NONE CONTRASTIVE + Online Clustering Base Approach'''
####################################################################################
'''
Why not using OFFLINE cluster method to assign the Psudo -- Group of lable feature on entire dataset
--> using this training in Supervised manner
----->  THIS method has to go through entire data to generate all these subset label (expensive to implement)
==> Need the Online Clustering for reduce the computation expensive of the method
# Using Non-Contrastive + Clustering Assignment
1. Building Prototype Net to Convert 256 Feature vector --> Smaller protype Vectors
2. Bulding Sinkhorn For clustering Assignment (SwAV-- Online assigment with Learnable parameter)
3. Cross entropy loss to optimize the encoder (2 version should be similar)
'''
# 1 Prototype Network


# 2 Building the SInkhorn cluster assignment with 3 Itertion
# Good number of iter for effcient runing in GPUs times

def sinkhorn(sample_prototype_batch):

    Q = tf.transpose(tf.exp(sample_prototype_batch/0.05))
    Q /= tf.keras.backend.sum(Q)
    K, B = Q.shape

    u = tf.zeros_like(K, dtype=tf.float32)
    r = tf.ones_like(K, dtype=tf.float32) / K
    c = tf.ones_like(B, dtype=tf.float32) / B

    for _ in range(3):
        u = tf.keras.backend.sum(Q, axis=1)
        Q *= tf.expand_dims((r / u), axis=1)
        Q *= tf.expand_dims(c / tf.keras.backend.sum(Q, axis=0), 0)

    final_quantity = Q / tf.keras.backend.sum(Q, axis=0, keepdims=True)
    final_quantity = tf.transpose(final_quantity)

    return final_quantity


def SwAV_loss(tape, crops_for_assign, batch_size, NUM_CROPS, prototype, temperature):
    loss = 0
    for i, crop_id in enumerate(crops_for_assign):
        with tape.stop_recording():
            out = prototype[batch_size * crop_id: batch_size*(crop_id+1)]
            # get assignment
            q = sinkhorn(out)

        # Cluster assigment predictiob
        subloss = 0
        for v in np.delete(np.arange(np.sum(NUM_CROPS)), crop_id):
            p = tf.nn.softmax(
                prototype[batch_size * v: batch_size * (v + 1)] / temperature)
            subloss -= tf.math.reduce_mean(
                tf.math.reduce_sum(q * tf.math.log(p), axis=1))
        loss += subloss / tf.cast((tf.reduce_sum(NUM_CROPS) - 1), tf.float32)

    loss /= len(crops_for_assign)

    return loss
