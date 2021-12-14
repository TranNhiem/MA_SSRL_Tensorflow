# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Training utilities."""

from absl import logging

import tensorflow as tf


def update_pretrain_metrics_train(contrast_loss, contrast_acc, contrast_entropy,
                                  loss, logits_con, labels_con):
    """Updated pretraining metrics."""
    #total contrastive_loss
    contrast_loss.update_state(loss)

    contrast_acc_val = tf.equal(
        tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1))
    contrast_acc_val = tf.reduce_mean(tf.cast(contrast_acc_val, tf.float32))
    
    #Contrastive acc
    contrast_acc.update_state(contrast_acc_val)

    prob_con = tf.nn.softmax(logits_con)
    entropy_con = -tf.reduce_mean(
        tf.reduce_sum(prob_con * tf.math.log(prob_con + 1e-8), -1))
    
    contrast_entropy.update_state(entropy_con)

def update_pretrain_metrics_eval(contrast_loss_metric,
                                 contrastive_top_1_accuracy_metric,
                                 contrastive_top_5_accuracy_metric,
                                 contrast_loss, logits_con, labels_con):
    contrast_loss_metric.update_state(contrast_loss)
    contrastive_top_1_accuracy_metric.update_state(
        tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1))
    contrastive_top_5_accuracy_metric.update_state(labels_con, logits_con)


def update_pretrain_binary_metrics_train_v0( contrast_binary_loss, contrast_bi_acc, contrast_bi_entroy, 
                                    bi_co_loss, logits_Obj, logits_Backg, labels_binary_con
                                  ):
    '''
    args: 

        bi_co_loss, Binary_contrastive_loss
        logits_Obj, object logits output
        logits_Backg, : background logits output
        labels_binary_con: is the label (Object and background feature) belong to

    Return: 
        Update result of these three metrics
        contrast_binary_loss, is total Binary contrast loss for each distributed run
        contrast_bi_acc, is the accuracy contrastive average result of Object and Backgroud
        contrast_bi_entroy is the entropy probability accuracy (average result) of object and backgroud

    '''
    
    # Binary contrast framework
    contrast_binary_loss.update_state(bi_co_loss)

    ## Calculate contrast binary accuracy (Object feature -- Background feature)
    #Object feature
    contrast_acc_obj = tf.equal(
        tf.argmax(labels_binary_con, 1), tf.argmax(logits_Obj, axis=1))
    contrast_acc_obj = tf.reduce_mean(tf.cast(contrast_acc_val, tf.float32))
    
    #background feature
    contrast_acc_backg=tf.equal(
        tf.argmax(labels_binary_con, 1), tf.argmax(logits_Backg, axis=1))
    contrast_acc_backg = tf.reduce_mean(tf.cast(contrast_acc_backg, tf.float32))

    total_contrast_acc= (contrast_acc_obj + contrast_acc_backg)/2
    
    contrast_bi_acc.update_state(contrast_acc_val)
    '''
    Noted consider update contrast_bi_acc BASE on Object Only
    '''
    #contrast_bi_acc.update_state(contrast_acc_obj)

    ## Calculate Contrast-Binary-Entropy
    #Object feature
    prob_con_obj = tf.nn.softmax(logits_Obj)
    entropy_con_Obj = -tf.reduce_mean(
        tf.reduce_sum(prob_con_obj * tf.math.log(prob_con_obj + 1e-8), -1))
    
    #backgroud feature
    prob_con_backg = tf.nn.softmax(logits_Backg)
    entropy_con_Backg = -tf.reduce_mean(
        tf.reduce_sum(prob_con_backg * tf.math.log(prob_con_backg + 1e-8), -1))
    all_entropy_prob=(entropy_con_Obj+entropy_con_Backg)/2
    
    contrast_bi_entroy.update_state(all_entropy_prob)
    '''
    Noted consider update contrast_bi_acc BASE on Object Only
    '''
    #contrast_bi_entroy.update_state(entropy_con_Obj)


def update_pretrain_binary_metrics_eval_v0(contrast_binary_loss_metric,
                                 contrastive_top_1_accuracy_metric,
                                 contrastive_top_5_accuracy_metric,
                                 contrast_binary_loss, logits_object, logits_backg, labels_con):
    
    '''
    args: 
     contrast_binary_loss: is the total loss sum up at the end 
     logits_object: is main object logits output 
     logits_backg: is the main background logits output  
     labels_con: is the lable for (Objects-backgroud logits) --> share the same lable Index

    Return 
        Update result for three metrics: 
        + contrastive_binary_loss 
        + contrastive_top_1 accuracy
        + contrastive_top_5 accuracy
    
    '''
    
    ## Total contrast_binary_loss each run
    contrast_binary_loss_metric.update_state(contrast_binary_loss)
    
    ## Contrastive accuracy of feature and background
    
    # Contrastive  accuracy for Object 
  
    object_top1= tf.argmax(labels_con, 1), tf.argmax(logits_object, axis=1)
    backgroud_top1= tf.argmax(labels_con, 1), tf.argmax(logits_backg, axis=1)
    all_top1=(object_top1+ backgroud_top1)/2
    
    '''
    Noted consider update contrast_bi_acc BASE on Object Only
    '''
    #contrastive_top_1_accuracy_metric.update(object_top1)

    contrastive_top_5_accuracy_metric.update_state(labels_con, logits_object)



def update_finetune_metrics_train(supervised_loss_metric, supervised_acc_metric,
                                  loss, labels, logits):
    supervised_loss_metric.update_state(loss)

    label_acc = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, axis=1))
    label_acc = tf.reduce_mean(tf.cast(label_acc, tf.float32))
    supervised_acc_metric.update_state(label_acc)


def update_finetune_metrics_eval(label_top_1_accuracy_metrics,
                                 label_top_5_accuracy_metrics, outputs, labels):
    label_top_1_accuracy_metrics.update_state(
        tf.argmax(labels, 1), tf.argmax(outputs, axis=1))
    label_top_5_accuracy_metrics.update_state(labels, outputs)


def _float_metric_value(metric):
    """Gets the value of a float-value keras metric."""
    return metric.result().numpy().astype(float)


def log_and_write_metrics_to_summary(all_metrics, global_step):
    for metric in all_metrics:
        metric_value = _float_metric_value(metric)
        logging.info('Step: [%d] %s = %f', global_step,
                     metric.name, metric_value)
        tf.summary.scalar(metric.name, metric_value, step=global_step)
