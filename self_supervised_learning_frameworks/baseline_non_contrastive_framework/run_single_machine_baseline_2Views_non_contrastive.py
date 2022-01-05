from losses_optimizers.learning_rate_optimizer import WarmUpAndCosineDecay, CosineAnnealingDecayRestarts
from objectives import metrics
from objectives import objective as obj_lib
from Neural_Net_Architecture.Convolution_Archs.ResNet_models import ssl_model as all_model
from losses_optimizers.self_supervised_losses import byol_loss
from config.helper_functions import *
from Augment_Data_utils.imagenet_dataloader_under_development import Imagenet_dataset
from tensorflow import distribute as tf_dis
import tensorflow as tf
import wandb
from tqdm import trange    # progress-bar presentation
import random
from absl import logging
import json
from math import ceil
import os
# for disable some tf warning message..
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# deep-learn pkgs
#       self-define pkgs


def main(FLAGS):
    # training sub_procedure
    # Scale loss  --> Aggregating all Gradients
    def distributed_loss(x1, x2):
        # each GPU loss per_replica batch loss
        per_example_loss, logits_ab, labels = byol_loss(
            x1, x2,  temperature=FLAGS.temperature)

        # total sum loss //Global batch_size
        loss = tf.reduce_sum(per_example_loss) * \
            (1./train_global_batch)
        return loss, logits_ab, labels

    @tf.function
    def distributed_train_step(ds_one, ds_two):
        per_replica_losses = strategy.run(
            train_step, args=(ds_one, ds_two))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)

    @tf.function
    def train_step(ds_one, ds_two):
        # Get the data from
        images_one, lable_one = ds_one
        images_two, lable_two = ds_two

        with tf.GradientTape(persistent=True) as tape:

            if FLAGS.non_contrast_loss == "byol_symmetrized_loss":
                logging.info("You implement Symmetrized loss")
                '''
                Symetrize the loss --> Need to switch image_1, image_2 to (Online -- Target Network)
                loss 1= L2_loss*[online_model(image1), target_model(image_2)]
                loss 2=  L2_loss*[online_model(image2), target_model(image_1)]
                symetrize_loss= (loss 1+ loss_2)/ 2

                '''

                # -------------------------------------------------------------
                # Passing image 1, image 2 to Online Encoder , Target Encoder
                # -------------------------------------------------------------

                # Online
                proj_head_output_1, supervised_head_output_1 = online_model(
                    images_one, training=True)
                proj_head_output_1 = prediction_model(
                    proj_head_output_1, training=True)

                # Target
                proj_head_output_2, supervised_head_output_2 = target_model(
                    images_two, training=True)

                # -------------------------------------------------------------
                # Passing Image 1, Image 2 to Target Encoder,  Online Encoder
                # -------------------------------------------------------------

                # online
                proj_head_output_2_online, _ = online_model(
                    images_two, training=True)
                # Vector Representation from Online encoder go into Projection head again
                proj_head_output_2_online = prediction_model(
                    proj_head_output_2_online, training=True)

                # Target
                proj_head_output_1_target, _ = target_model(
                    images_one, training=True)

                # Compute Contrastive Train Loss -->
                loss = None
                if proj_head_output_1 is not None:
                    # Compute Contrastive Loss model
                    # Loss of the image 1, 2 --> Online, Target Encoder
                    loss_1_2, logits_ab, labels = distributed_loss(
                        proj_head_output_1, proj_head_output_2)

                    # Loss of the image 2, 1 --> Online, Target Encoder
                    loss_2_1, logits_ab_2, labels_2 = distributed_loss(
                        proj_head_output_2_online, proj_head_output_1_target)

                    # symetrized loss
                    loss = (loss_1_2 + loss_2_1)/2

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

            elif FLAGS.non_contrast_loss == "byol_asymmetrized_loss":
                logging.info("You implement Asymmetrized loss")
                # -------------------------------------------------------------
                # Passing image 1, image 2 to Online Encoder , Target Encoder
                # -------------------------------------------------------------

                # Online
                proj_head_output_1, supervised_head_output_1 = online_model(
                    images_one, training=True)
                proj_head_output_1 = prediction_model(
                    proj_head_output_1, training=True)

                # Target
                proj_head_output_2, supervised_head_output_2 = target_model(
                    images_two, training=True)

                # Compute Contrastive Train Loss -->
                loss = None
                if proj_head_output_1 is not None:
                    # Compute Contrastive Loss model
                    # Loss of the image 1, 2 --> Online, Target Encoder
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

            else:
                raise ValueError("Invalid Type loss")
            # Compute the Supervised train Loss
            '''Consider Sperate Supervised Loss'''
            # supervised_loss=None
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
                    # scale_sup_loss = tf.reduce_sum(
                    #     sup_loss) * (1./train_global_batch)
                    # Update Supervised Metrics
                    metrics.update_finetune_metrics_train(supervised_loss_metric,
                                                          supervised_acc_metric, scale_sup_loss,
                                                          supervise_lable, outputs)

                '''Attention'''
                # Noted Consideration Aggregate (Supervised + Contrastive Loss) --> Update the Model Gradient
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

            weight_decay_loss = all_model.add_weight_decay(
                online_model, adjust_per_optimizer=True)
            # Under experiment Scale loss after adding Regularization and scaled by Batch_size
            # weight_decay_loss = tf.nn.scale_regularization_loss(
            #     weight_decay_loss)
            weight_decay_metric.update_state(weight_decay_loss)
            loss += weight_decay_loss

            total_loss_metric.update_state(loss)

            logging.info('Trainable variables:')
            for var in online_model.trainable_variables:
                logging.info(var.name)
            # ----------------------------------------------
            # This section for Mixed Precision Training
            # ----------------------------------------------

        if FLAGS.mixprecision == "fp16":
            logging.info("you implement mix_percision_16_Fp")

            # Reduce loss Precision to 16 Bits
            # Method 1
            scaled_loss = optimizer.get_scaled_loss(loss)
            # Update the Encoder
            scaled_gradients = tape.gradient(
                scaled_loss, online_model.trainable_variables)
            gradients = optimizer.get_unscaled_gradients(scaled_gradients)
            optimizer.apply_gradients(
                zip(gradients, online_model.trainable_variables))

            # Update Prediction Head model
            scaled_grads = tape.gradient(
                scaled_loss, prediction_model.trainable_variables)
            gradients_unscale = optimizer.get_unscaled_gradients(scaled_grads)
            optimizer.apply_gradients(
                zip(gradients_unscale, prediction_model.trainable_variables))

            # Method 2
            # fp32_grads = tape.gradient(loss, online_model.trainable_variables)
            # fp16_grads = [tf.cast(grad, 'float16')for grad in fp32_grads]
            # all_reduce_fp16_grads = tf.distribute.get_replica_context(
            # ).all_reduce(tf.distribute.ReduceOp.SUM, fp16_grads)
            # all_reduce_fp32_grads = [
            #     tf.cast(grad, 'float32')for grad in all_reduce_fp16_grads]

            # all_reduce_fp32_grads = optimizer.get_unscaled_gradients(
            #     all_reduce_fp32_grads)
            # optimizer.apply_gradients(zip(
            #     all_reduce_fp32_grads, online_model.trainable_variables), experimental_aggregate_gradients=False)

            # # Method 2
            # fp32_grads = tape.gradient(
            #     loss, prediction_model.trainable_variables)
            # fp16_grads = [tf.cast(grad, 'float16')for grad in fp32_grads]
            # all_reduce_fp16_grads = tf.distribute.get_replica_context(
            # ).all_reduce(tf.distribute.ReduceOp.SUM, fp16_grads)
            # all_reduce_fp32_grads = [
            #     tf.cast(grad, 'float32')for grad in all_reduce_fp16_grads]
            # all_reduce_fp32_grads = optimizer.get_unscaled_gradients(
            #     all_reduce_fp32_grads)
            # optimizer.apply_gradients(zip(
            #     all_reduce_fp32_grads, prediction_model.trainable_variables), experimental_aggregate_gradients=False)

        elif FLAGS.mixprecision == "fp32":
            logging.info("you implement original_Fp precision")

            # Update Encoder and Projection head weight
            grads = tape.gradient(loss, online_model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, online_model.trainable_variables))

            # Update Prediction Head model
            grads = tape.gradient(
                loss, prediction_model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, prediction_model.trainable_variables))
        else:
            raise ValueError(
                "Invalid Implement optimization floating precision")

        del tape
        return loss

    # 1. Prepare imagenet dataset
    strategy = tf_dis.MirroredStrategy()
    train_global_batch = FLAGS.train_global_batch = FLAGS.train_batch_size * \
        strategy.num_replicas_in_sync
    val_global_batch = FLAGS.val_global_batch = FLAGS.val_batch_size * \
        strategy.num_replicas_in_sync

    ds_args = {'img_size': FLAGS.image_size, 'train_path': FLAGS.train_path, 'val_path': FLAGS.val_path,
               'train_label': FLAGS.train_label, 'val_label': FLAGS.val_label, 'subset_class_num': FLAGS.num_classes,
               'train_batch': train_global_batch, 'val_batch': val_global_batch, 'strategy': strategy}
    train_dataset = Imagenet_dataset(**ds_args)

    #   baseline simclr style data augmentation
    train_ds = train_dataset.simclr_crop_da("rand_glb")
    #   performing Linear-protocol
    val_ds = train_dataset.supervised_validation()

    #   calculate the training related meta-info
    n_tra_sample, n_evl_sample = train_dataset.get_data_size()
    train_steps = FLAGS.eval_steps or int(
        n_tra_sample * FLAGS.train_epochs // train_global_batch)*2
    # is that necessary to further convert into int ?
    eval_steps = FLAGS.eval_steps or ceil(n_evl_sample / val_global_batch)
    # is that necessary to further convert into int ?
    epoch_steps = round(n_tra_sample / train_global_batch)

    checkpoint_steps = (FLAGS.checkpoint_steps or (
        FLAGS.checkpoint_epochs * epoch_steps))

    logging.info(f"# Subset_training class {FLAGS.num_classes}")
    logging.info(f"# train examples: {n_tra_sample}")
    logging.info(f"# train_steps: {train_steps}")
    logging.info(f"# eval examples: {n_evl_sample}")
    logging.info(f"# eval steps: {eval_steps}")

    # record the config in wanda database..
    wandb_record()

    # 2. Configure the Encoder Architecture
    with strategy.scope():
        online_model = all_model.online_model(FLAGS.num_classes)
        prediction_model = all_model.prediction_head_model()
        target_model = all_model.online_model(FLAGS.num_classes)

    # 3. perform the train/eval loop
    #   (1) training framework :
    if "train" in FLAGS.mode:
        summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)
        with strategy.scope():
            # Configure the learning rate
            if FLAGS.lr_strategies == "warmup_cos_lr":
                base_lr = FLAGS.base_lr
                scale_lr = FLAGS.lr_rate_scaling
                warmup_epochs = FLAGS.warmup_epochs
                train_epochs = FLAGS.train_epochs

                lr_schedule = WarmUpAndCosineDecay(
                    base_lr, train_global_batch, n_tra_sample, scale_lr, warmup_epochs,
                    train_epochs=train_epochs, train_steps=train_steps)

            elif FLAGS.lr_strategies == "cos_annealing_restart":
                base_lr = FLAGS.base_lr
                scale_lr = FLAGS.lr_rate_scaling
                # Control cycle of next step base of Previous step (2 times more steps)
                t_mul = 2.0
                # Control ititial Learning Rate Values (Next step equal to previous steps)
                m_mul = 1.0
                alpha = 0.0  # Final values of learning rate
                first_decay_steps = train_steps / (FLAGS.number_cycles * t_mul)
                lr_schedule = CosineAnnealingDecayRestarts(
                    base_lr, first_decay_steps, train_global_batch, scale_lr, t_mul=t_mul, m_mul=m_mul, alpha=alpha)

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
                    'train/non_contrast_loss')
                contrast_acc_metric = tf.keras.metrics.Mean(
                    "train/non_contrast_acc")
                contrast_entropy_metric = tf.keras.metrics.Mean(
                    'train/non_contrast_entropy')
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
                online_model, optimizer.iterations, optimizer)

            global_step = optimizer.iterations
            for epoch in trange(FLAGS.train_epochs):

                total_loss = 0.0
                num_batches = 0

                for _, (ds_one, ds_two) in enumerate(train_ds):

                    total_loss += distributed_train_step(ds_one, ds_two)
                    num_batches += 1

                    # Update weight of Target Encoder Every Step
                    beta = 0.99
                    target_encoder_weights = target_model.get_weights()
                    online_encoder_weights = online_model.get_weights()

                    for i in range(len(online_encoder_weights)):
                        target_encoder_weights[i] = beta * target_encoder_weights[i] + (
                            1-beta) * online_encoder_weights[i]
                    target_model.set_weights(target_encoder_weights)

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
                # Saving Entire Model
                if (epoch+1) % 20 == 0:
                    save_encoder = os.path.join(
                        FLAGS.model_dir, "encoder_model_" + str(epoch) + ".h5")
                    save_online_model = os.path.join(
                        FLAGS.model_dir, "online_model_" + str(epoch) + ".h5")
                    save_target_model = os.path.join(
                        FLAGS.model_dir, "target_model_" + str(epoch) + ".h5")

                    online_model.resnet_model.save_weights(save_encoder)
                    online_model.save_weights(save_online_model)
                    target_model.save_weights(save_target_model)
            logging.info('Training Complete ...')

    #   (2) evaluation framework :
    if "eval" in FLAGS.mode:
        ckpt_iter = [checkpoint_manager.latest_checkpoint] if "train" not in FLAGS.mode \
            else tf.train.checkpoints_iterator(FLAGS.model_dir, min_interval_secs=15)
        for ckpt in ckpt_iter:
            perform_evaluation(online_model, val_ds,
                               eval_steps, ckpt, strategy)

            # global_step from ckpt
            if result['global_step'] >= train_steps:
                logging.info('Evaluation complete. Existing-->')


# Utils function
# Setting GPU
def set_gpu_env(n_gpus=8):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[0:n_gpus], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,",
                  len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)

# Configure Wandb Training & for Weight and Bias Tracking Experiment


def wandb_record():
    configs = {
        "Model_Arch": "ResNet50",
        "Training mode": "Baseline Non_Contrastive",
        "DataAugmentation_types": "SimCLR_Inception_style_Croping",
        "Dataset": "ImageNet1k",

        "IMG_SIZE": FLAGS.image_size,
        "Epochs": FLAGS.train_epochs,
        "Batch_size": FLAGS.train_global_batch,
        "Learning_rate": FLAGS.base_lr,
        "Temperature": FLAGS.temperature,
        "Optimizer": FLAGS.optimizer,
        "SEED": FLAGS.SEED,
        "Subset_dataset": FLAGS.num_classes,
        "Loss type": FLAGS.aggregate_loss,
        "opt": FLAGS.up_scale
    }
    wandb.init(project=FLAGS.wandb_project_name, name=FLAGS.wandb_run_name, mode=FLAGS.wandb_mod,
               sync_tensorboard=True, config=configs)


if __name__ == '__main__':
    from config.config_non_contrast import read_cfg
    flag = read_cfg()
    set_gpu_env()

    main(flag.FLAGS)
