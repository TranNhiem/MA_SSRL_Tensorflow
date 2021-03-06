from config.absl_mock import Mock_Flag
from config.config_v0 import read_cfg
import os
import json
import math
import wandb
import random
from absl import flags
from absl import logging
from absl import app

import tensorflow as tf
from loss_optimizers.learning_rate_optimizer import WarmUpAndCosineDecay
import objectives.metrics as metrics
from Augment_Data_utils.imagenet_dataloader_under_development import Imagenet_dataset
from loss_optimizers.self_supervised_losses import nt_xent_symetrize_loss_simcrl, nt_xent_asymetrize_loss_v2
import model as all_model
import objectives.objective as obj_lib
from imutils import paths
from config.helper_functions import *
# Setting GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0:8], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

read_cfg()
flag = Mock_Flag()
FLAGS = flag.FLAGS


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Preparing dataset
    # Imagenet path prepare localy

    strategy = tf.distribute.MirroredStrategy()
    train_global_batch = FLAGS.train_batch_size * strategy.num_replicas_in_sync
    val_global_batch = FLAGS.val_batch_size * strategy.num_replicas_in_sync

    train_dataset = Imagenet_dataset(img_size, train_batch, val_batch)(img_size=FLAGS.image_size, train_batch=train_global_batch,  val_batch=val_global_batch,
                                                                       strategy=strategy, train_path=FLAGS.train_path, val_path=FLAGS.val_path,
                                                                       train_label=FLAGS.train_label, val_label=FLAGS.val_label,
                                                                       mask_path=FLAGS.mask_path, bi_mask=False)

    train_ds = train_dataset.simclr_random_global_crop()

    val_ds = train_dataset.supervised_validation()

    num_train_examples, num_eval_examples = train_dataset.get_data_size()

    train_steps = FLAGS.eval_steps or int(
        num_train_examples * FLAGS.train_epochs // train_global_batch)*2
    eval_steps = FLAGS.eval_steps or int(
        math.ceil(num_eval_examples / val_global_batch))

    epoch_steps = int(round(num_train_examples / train_global_batch))
    checkpoint_steps = (FLAGS.checkpoint_steps or (
        FLAGS.checkpoint_epochs * epoch_steps))

    logging.info('# train examples: %d', num_train_examples)
    logging.info('# train_steps: %d', train_steps)
    logging.info('# eval examples: %d', num_eval_examples)
    logging.info('# eval steps: %d', eval_steps)

    # Configure the Encoder Architecture.
    with strategy.scope():
        model = all_model.Model(FLAGS.num_classes)

    # Configure Wandb Training
    # Weight&Bias Tracking Experiment
    configs = {
        "Model_Arch": "ResNet50",
        "Training mode": "SSL",
        "DataAugmentation_types": "SimCLR_Inception_style_Croping",
        "Dataset": "ImageNet1k",

        "IMG_SIZE": FLAGS.image_size,
        "Epochs": FLAGS.train_epochs,
        "Batch_size": train_global_batch,
        "Learning_rate": FLAGS.base_lr,
        "Temperature": FLAGS.temperature,
        "Optimizer": FLAGS.optimizer,
        "SEED": FLAGS.SEED,
        "Loss type": FLAGS.aggregate_loss,
        "loss_version": FLAGS.loss_options
    }

    wandb.init(project="heuristic_attention_representation_learning",
               sync_tensorboard=True, config=configs)

    # Training Configuration
    # *****************************************************************
    # Only Evaluate model
    # *****************************************************************

    if FLAGS.mode == "eval":
        # can choose different min_interval
        for ckpt in tf.train.checkpoints_iterator(FLAGS.model_dir, min_interval_secs=15):
            result = perform_evaluation(
                model, val_ds, eval_steps, ckpt, strategy)
            # global_step from ckpt
            if result['global_step'] >= train_steps:
                logging.info('Evaluation complete. Existing-->')

    # *****************************************************************
    # Pre-Training and Evaluate
    # *****************************************************************
    else:
        summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)

        with strategy.scope():

            # Configure the learning rate
            base_lr = FLAGS.base_lr
            scale_lr = FLAGS.lr_rate_scaling
            warmup_epochs = FLAGS.warmup_epochs
            train_epochs = FLAGS.train_epochs
            lr_schedule = WarmUpAndCosineDecay(
                base_lr, train_global_batch, num_train_examples, scale_lr, warmup_epochs,
                train_epochs=train_epochs, train_steps=train_steps)

            # Current Implement the Mixpercision optimizer
            optimizer = all_model.build_optimizer(lr_schedule)

            # Build tracking metrics
            all_metrics = []
            # Linear classfiy metric
            weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
            total_loss_metric = tf.keras.metrics.Mean('train/total_loss')
            all_metrics.extend([weight_decay_metric, total_loss_metric])

            if FLAGS.train_mode == 'pretrain':
                # for contrastive metrics
                contrast_loss_metric = tf.keras.metrics.Mean(
                    'train/contrast_loss')
                contrast_acc_metric = tf.keras.metrics.Mean(
                    "train/contrast_acc")
                contrast_entropy_metric = tf.keras.metrics.Mean(
                    'train/contrast_entropy')
                all_metrics.extend(
                    [contrast_loss_metric, contrast_acc_metric, contrast_entropy_metric])

            if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
                logging.info(
                    "Apllying pre-training and Linear evaluation at the same time")
                # Fine-tune architecture metrics
                supervised_loss_metric = tf.keras.metrics.Mean(
                    'train/supervised_loss')
                supervised_acc_metric = tf.keras.metrics.Mean(
                    'train/supervised_acc')
                all_metrics.extend(
                    [supervised_loss_metric, supervised_acc_metric])

            # Check and restore Ckpt if it available
            # Restore checkpoint if available.
            checkpoint_manager = try_restore_from_checkpoint(
                model, optimizer.iterations, optimizer)

            # Scale loss  --> Aggregating all Gradients
            def distributed_loss(x1, x2):

                if FLAGS.loss_options == "loss_v0":
                    # each GPU loss per_replica batch loss
                    per_example_loss, logits_ab, labels = nt_xent_symetrize_loss_simcrl(
                        x1, x2, LARGE_NUM=FLAGS.LARGE_NUM, hidden_norm=FLAGS.hidden_norm, temperature=FLAGS.temperature)

                elif FLAGS.loss_options == "loss_v1":
                    # each GPU loss per_replica batch loss
                    x_1_2 = tf.concat([x1, x2], axis=0)
                    per_example_loss, logits_ab, labels = nt_xent_asymetrize_loss_v2(
                        x_1_2,  temperature=FLAGS.temperature)

                else:
                    raise ValueError("Loss version is not implement yet")

                # total sum loss //Global batch_size
                loss = tf.reduce_sum(per_example_loss) * \
                    (1./train_global_batch)
                return loss, logits_ab, labels

            @tf.function
            def train_step(ds_one, ds_two):
                # Get the data from
                images_one, lable_one = ds_one
                images_two, lable_two = ds_two

                with tf.GradientTape() as tape:

                    print(images_one)
                    proj_head_output_1, supervised_head_output_1 = model(
                        images_one, training=True)
                    proj_head_output_2, supervised_head_output_2 = model(
                        images_two, training=True)

                    # Compute Contrastive Train Loss -->
                    loss = None
                    if proj_head_output_1 is not None:
                        # Compute Contrastive Loss model
                        loss, logits_ab, labels = distributed_loss(
                            proj_head_output_1, proj_head_output_2)

                        if loss is None:
                            loss = loss
                        else:
                            loss += loss
                        # Update Self-Supervised Metrics
                        metrics.update_pretrain_metrics_train(contrast_loss_metric,
                                                              contrast_acc_metric,
                                                              contrast_entropy_metric,
                                                              loss, logits_ab,
                                                              labels)
                    # Compute the Supervised train Loss
                    if supervised_head_output_1 is not None:

                        if FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:

                            outputs = tf.concat(
                                [supervised_head_output_1, supervised_head_output_2], 0)
                            supervise_lable = tf.concat(
                                [lable_one, lable_two], 0)

                            # Calculte the cross_entropy loss with Labels
                            sup_loss = obj_lib.add_supervised_loss(
                                labels=supervise_lable, logits=outputs)

                            scale_sup_loss = tf.nn.compute_average_loss(
                                sup_loss, global_batch_size=train_global_batch)
                            # scale_sup_loss =  tf.reduce_sum(sup_loss) * (1./train_global_batch)
                            # Update Supervised Metrics
                            metrics.update_finetune_metrics_train(supervised_loss_metric,
                                                                  supervised_acc_metric, scale_sup_loss,
                                                                  supervise_lable, outputs)

                        '''Attention'''
                        # Noted Consideration Aggregate (Supervised + Contrastive Loss)
                        # --> Update the Model Gradient base on Loss
                        # Option 1: Only use Contrast loss
                        # option 2: Contrast Loss + Supervised Loss
                        if FLAGS.aggregate_loss == "contrastive_supervised":

                            if loss is None:
                                loss = scale_sup_loss
                            else:
                                loss += scale_sup_loss

                        elif FLAGS.aggregate_loss == "contrastive":

                            supervise_loss = None
                            if supervise_loss is None:
                                supervise_loss = scale_sup_loss
                            else:
                                supervise_loss += scale_sup_loss
                        else:
                            raise ValueError(
                                " Loss aggregate is invalid please check FLAGS.aggregate_loss")

                    # Consideration Remove L2 Regularization Loss
                    # --> This Only Use for Supervised Head
                    weight_decay_loss = all_model.add_weight_decay(
                        model, adjust_per_optimizer=True)
                   # Under experiment Scale loss after adding Regularization and scaled by Batch_size
                    # weight_decay_loss = tf.nn.scale_regularization_loss(
                    #     weight_decay_loss)

                    weight_decay_metric.update_state(weight_decay_loss)

                    loss += weight_decay_loss
                    # Contrast loss + Supervised loss + Regularize loss
                    total_loss_metric.update_state(loss)

                    logging.info('Trainable variables:')
                    for var in model.trainable_variables:
                        logging.info(var.name)

                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(grads, model.trainable_variables))

                return loss

            @tf.function
            def distributed_train_step(ds_one, ds_two):
                per_replica_losses = strategy.run(
                    train_step, args=(ds_one, ds_two))
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                       axis=None)

            global_step = optimizer.iterations

            for epoch in range(FLAGS.train_epochs):

                total_loss = 0.0
                num_batches = 0
                for _, (ds_one, ds_two) in enumerate(train_ds):

                    total_loss += distributed_train_step(ds_one, ds_two)
                    num_batches += 1
                    # if (global_step.numpy()+ 1) % checkpoint_steps==0:
                    with summary_writer.as_default():
                        cur_step = global_step.numpy()
                        checkpoint_manager.save(cur_step)
                        logging.info('Completed: %d / %d steps',
                                     cur_step, train_steps)
                        metrics.log_and_write_metrics_to_summary(
                            all_metrics, cur_step)
                        tf.summary.scalar('learning_rate', lr_schedule(tf.cast(global_step, dtype=tf.float32)),
                                          global_step)
                        summary_writer.flush()

                epoch_loss = total_loss/num_batches
                # Wandb Configure for Visualize the Model Training
                wandb.log({
                    "epochs": epoch+1,
                    "train_contrast_loss": contrast_loss_metric.result(),
                    "train_contrast_acc": contrast_acc_metric.result(),
                    "train_contrast_acc_entropy": contrast_entropy_metric.result(),
                    "train/weight_decay": weight_decay_metric.result(),
                    "train/total_loss": epoch_loss,
                    "train/supervised_loss":    supervised_loss_metric.result(),
                    "train/supervised_acc": supervised_acc_metric.result()
                })
                for metric in all_metrics:
                    metric.reset_states()
                if epoch+1 == 50:
                    save_ = './model_ckpt/resnet_simclr/baseline_encoder_resnet50_mlp' + \
                        str(epoch) + ".h5"
                    model.save_weights(save_)
            logging.info('Training Complete ...')

        if FLAGS.mode == 'train_then_eval':
            perform_evaluation(model, val_ds, eval_steps,
                               checkpoint_manager.latest_checkpoint, strategy)

    # Pre-Training and Finetune
if __name__ == '__main__':

    app.run(main)
