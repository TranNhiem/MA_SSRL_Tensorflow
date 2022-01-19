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
from math import cos, pi
from tqdm import trange    # progress-bar presentation
import random
from absl import logging
import json
from math import ceil
import os

# For setting GPUs Thread reduce kernel Luanch Delay
# https://github.com/tensorflow/tensorflow/issues/25724
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '2'

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

class Runner(object):
    def __init__(self, FLAGS, wanda_cfg=None):

        # Configure Wandb Training & for Weight and Bias Tracking Experiment
        def wandb_init(wanda_cfg):
            if wanda_cfg:
                wandb.init(project=self.wandb_project_name, name=FLAGS.wandb_run_name, mode=self.wandb_mod,
                           sync_tensorboard=True, config=wanda_cfg)
        
        #   calculate the training related meta-info
        def infer_ds_info(n_tra_sample, n_evl_sample, train_global_batch, val_global_batch):
            self.train_steps = self.eval_steps or int(
                n_tra_sample * self.train_epochs // train_global_batch)*2
            self.epoch_steps = round(n_tra_sample / train_global_batch)
            eval_steps = self.eval_steps or ceil(
                n_evl_sample / val_global_batch)

            # logging the ds info
            logging.info(f"# Subset_training class {self.num_classes}")
            logging.info(f"# train examples: {n_tra_sample}")
            logging.info(f"# eval examples: {n_evl_sample}")
            logging.info(f"# train_steps: {self.train_steps}")
            logging.info(f"# eval steps: {self.eval_steps}")

        # we use 'self' to access config in this class.. (not good, but i take it)
        self.__dict__ = FLAGS.__dict__

        # 1. Prepare imagenet dataset
        strategy = tf_dis.MirroredStrategy()
        self.strategy = strategy
        train_global_batch = self.train_batch_size * strategy.num_replicas_in_sync
        val_global_batch = self.val_batch_size * strategy.num_replicas_in_sync
        ds_args = {'img_size': self.image_size, 'train_path': self.train_path, 'val_path': self.val_path,
                   'train_label': self.train_label, 'val_label': self.val_label, 'subset_class_num': self.num_classes,
                   'train_batch': train_global_batch, 'val_batch': val_global_batch, 'strategy': strategy, 'seed':self.SEED}
        # Dataloader V2 already be proposed as formal data_loader
        train_dataset = Imagenet_dataset(**ds_args)

        n_tra_sample, n_evl_sample = train_dataset.get_data_size()
        infer_ds_info(n_tra_sample, n_evl_sample,
                      train_global_batch, val_global_batch)

        # initial record utils
        wanda_cfg['Batch_size'] = train_global_batch
        wandb_init(wanda_cfg)
        self.summary_writer = tf.summary.create_file_writer(self.model_dir)

        checkpoint_steps = (self.checkpoint_steps or (
            self.checkpoint_epochs * self.epoch_steps))

        # record var into self
        self.strategy = strategy
        self.train_global_batch, self.val_global_batch = train_global_batch, val_global_batch
        self.n_tra_sample = n_tra_sample
        self.train_dataset = train_dataset

    def train(self, exe_mode, da_crp_key="rnd_crp"):
        # Configure the Encoder Architecture
        def get_gpu_model():
            with self.strategy.scope():
                online_model = all_model.online_model(self.num_classes)
                prediction_model = all_model.prediction_head_model()
                target_model = all_model.online_model(self.num_classes)
            return online_model, prediction_model, target_model

        def get_optimizer():
            with self.strategy.scope():
                # Configure the learning rate
                if self.lr_strategies == "warmup_cos_lr":
                    base_lr = self.base_lr
                    scale_lr = self.lr_rate_scaling
                    warmup_epochs = self.warmup_epochs
                    train_epochs = self.train_epochs

                    lr_schedule = WarmUpAndCosineDecay(
                        base_lr, self.train_global_batch, self.n_tra_sample, scale_lr, warmup_epochs,
                        train_epochs=train_epochs, train_steps=self.train_steps)

                elif self.lr_strategies == "cos_annealing_restart":
                    base_lr = self.base_lr
                    scale_lr = self.lr_rate_scaling
                    # Control cycle of next step base of Previous step (2 times more steps)
                    t_mul = 2.0
                    # Control ititial Learning Rate Values (Next step equal to previous steps)
                    m_mul = 1.0
                    alpha = 0.0  # Final values of learning rate
                    first_decay_steps = self.train_steps / \
                        (self.number_cycles * t_mul)
                    lr_schedule = CosineAnnealingDecayRestarts(
                        base_lr, first_decay_steps, self.train_global_batch, scale_lr, t_mul=t_mul, m_mul=m_mul, alpha=alpha)

                # Current Implement the Mixpercision optimizer
                optimizer = all_model.build_optimizer(lr_schedule)
            return lr_schedule, optimizer

        def get_metrics():
            with self.strategy.scope():
                # Build tracking metrics
                metric_dict = {}
                # Linear classfiy metric
                metric_dict['weight_decay_metric'] = tf.keras.metrics.Mean(
                    'train/weight_decay')
                metric_dict['total_loss_metric'] = tf.keras.metrics.Mean(
                    'train/total_loss')

                if self.train_mode == 'pretrain':
                    # for contrastive metrics
                    metric_dict['contrast_loss_metric'] = tf.keras.metrics.Mean(
                        'train/non_contrast_loss')
                    metric_dict['contrast_acc_metric'] = tf.keras.metrics.Mean(
                        "train/non_contrast_acc")
                    metric_dict['contrast_entropy_metric'] = tf.keras.metrics.Mean(
                        'train/non_contrast_entropy')

                if self.train_mode == 'finetune' or self.lineareval_while_pretraining:
                    logging.info(
                        "Apllying pre-training and Linear evaluation at the same time")
                    # Fine-tune architecture metrics
                    metric_dict['supervised_loss_metric'] = tf.keras.metrics.Mean(
                        'train/supervised_loss')
                    metric_dict['supervised_acc_metric'] = tf.keras.metrics.Mean(
                        'train/supervised_acc')
            return metric_dict

        def log_wandb(epoch, epoch_loss, metric_dict):
            # Wandb Configure for Visualize the Model Training
            wandb.log({
                "epochs": epoch+1,
                "train_contrast_loss": metric_dict['contrast_loss_metric'].result(),
                "train_contrast_acc": metric_dict['contrast_acc_metric'].result(),
                "train_contrast_acc_entropy": metric_dict['contrast_entropy_metric'].result(),
                "train/weight_decay": metric_dict['weight_decay_metric'].result(),
                "train/total_loss": epoch_loss,
                "train/supervised_loss": metric_dict['supervised_loss_metric'].result(),
                "train/supervised_acc": metric_dict['supervised_acc_metric'].result()
            })

        # prepare train related obj
        self.online_model, self.prediction_model, self.target_model = get_gpu_model()
        # assign to self.opt to prevent the namespace covered
        lr_schedule, optimizer = _, self.opt = get_optimizer()
        self.metric_dict = metric_dict = get_metrics()

        # perform data_augmentation by calling the dataloader methods
        train_ds = self.train_dataset.multi_view_data_aug(self.train_dataset.Fast_Augment, 
                                                            policy_type="imagenet")

        ##   performing Linear-protocol
        val_ds = self.train_dataset.supervised_validation()

        # Check and restore Ckpt if it available
        # Restore checkpoint if available.
        checkpoint_manager = try_restore_from_checkpoint(
            self.online_model, optimizer.iterations, optimizer)

        global_step = optimizer.iterations
        for epoch in trange(self.train_epochs):

            total_loss = 0.0
            num_batches = 0

            ## batch_size, ((data, lab), (data, lab))
            #      2 global view vs. 3 local view
            for _, ds_pkgs in enumerate(train_ds):
                total_loss += self.__distributed_train_step(ds_pkgs)
                num_batches += 1

                # Update weight of Target Encoder Every Step
                if FLAGS.moving_average == "fixed_value":
                    beta = 0.99
                if FLAGS.moving_average == "schedule":
                    # This update the Beta value schedule along with Trainign steps Follow BYOL
                    beta_base = 0.996
                    cur_step = global_step.numpy()
                    beta = 1 - (1-beta_base) * \
                        (cos(pi*cur_step/self.train_steps)+1)/2

                target_model, online_model = self.target_model, self.online_model
                target_encoder_weights = target_model.get_weights()
                online_encoder_weights = online_model.get_weights()
                # mean teacher update
                for lay_idx in range(len(online_encoder_weights)):
                    target_encoder_weights[lay_idx] = beta * target_encoder_weights[lay_idx] + (
                        1-beta) * online_encoder_weights[lay_idx]
                target_model.set_weights(target_encoder_weights)

                with self.summary_writer.as_default():
                    cur_step = global_step.numpy()
                    checkpoint_manager.save(cur_step)
                    logging.info('Completed: %d / %d steps',
                                 cur_step, self.train_steps)
                    all_metrics = list(metric_dict.values())
                    metrics.log_and_write_metrics_to_summary(
                        all_metrics, cur_step)
                    tf.summary.scalar('learning_rate', lr_schedule(tf.cast(global_step, dtype=tf.float32)),
                                      global_step)
                    self.summary_writer.flush()

            epoch_loss = total_loss/num_batches
            log_wandb(epoch, epoch_loss, metric_dict)
            for metric in metric_dict.values():
                metric.reset_states()

            # perform_evaluation(self.online_model, val_ds, eval_steps,
            #                    checkpoint_manager.latest_checkpoint, self.strategy)

            # Saving Entire Model
            if (epoch+1) % 20 == 0:
                save_encoder = os.path.join(
                    self.model_dir, f"encoder_model_{epoch}.h5")
                save_online_model = os.path.join(
                    self.model_dir, f"online_model_{epoch}.h5")
                save_target_model = os.path.join(
                    self.model_dir, f"target_model_{epoch}.h5")
                self.online_model.resnet_model.save_weights(save_encoder)
                self.online_model.save_weights(save_online_model)
                self.target_model.save_weights(save_target_model)
            logging.info('Training Complete ...')

            if (epoch + 1) % 20 == 0:
                FLAGS.train_mode = 'finetune'
                result = perform_evaluation(self.online_model, val_ds, self.eval_steps,
                                            checkpoint_manager.latest_checkpoint, self.strategy)
                wandb.log({
                    "eval/label_top_1_accuracy": result["eval/label_top_1_accuracy"],
                    "eval/label_top_5_accuracy": result["eval/label_top_5_accuracy"],
                })
                FLAGS.train_mode = 'pretrain'
        # perform eval after training
        if "eval" in exe_mode:
            self.eval(checkpoint_manager)

    def eval(self, checkpoint_manager=None):
        ckpt_iter = [checkpoint_manager.latest_checkpoint] if "train" not in self.mode \
            else tf.train.checkpoints_iterator(self.model_dir, min_interval_secs=15)
        for ckpt in ckpt_iter:
            perform_evaluation(self.online_model, val_ds,
                               eval_steps, ckpt, strategy)

            # global_step from ckpt
            if result['global_step'] >= train_steps:
                logging.info('Evaluation complete. Existing-->')

    # Training sub_procedure :
    @tf.function
    def __distributed_train_step(self, ds_pkgs):
        per_replica_losses = self.strategy.run(
            self.__train_step, args=(ds_pkgs))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                    axis=None)

    @tf.function
    def __train_step(self, ds_pkgs):
        # Scale loss  --> Aggregating all Gradients
        def distributed_loss(x1, x2, x3, x4, x5):
            # each GPU loss per_replica batch loss
            per_example_loss_1, logits_ab, labels = byol_loss(
                x1, x2,  temperature=self.temperature)

            per_example_loss_2, _, _ = byol_loss(
                x3, x4,  temperature=self.temperature)

            per_example_loss_3, _, _ = byol_loss(
                x3, x5,  temperature=self.temperature)

            per_example_loss = per_example_loss_1 + per_example_loss_2 + per_example_loss_3

            # total sum loss //Global batch_size
            loss = tf.reduce_sum(per_example_loss) * \
                (1./self.train_global_batch)

            return loss, logits_ab, labels

        # Unpacking the data:
        #   only performs supervised on global view 
        ds_1, lable_one = ds_pkgs[0]
        ds_2, lable_two = ds_pkgs[1]
        ds_3, _ = ds_pkgs[2]
        ds_4, _ = ds_pkgs[3]
        ds_5, _ = ds_pkgs[4]

        with tf.GradientTape(persistent=True) as tape:

            if FLAGS.loss_type == "byol_symmetrized_loss":
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
                if FLAGS.XLA_compiler == "model_only":
                    logging.info("XLA Compiler for model")
                    with tf.xla.experimental.jit_scope():
                        # Online
                        proj_head_output_1, supervised_head_output_1 = self.online_model(
                            images_one, training=True)
                        proj_head_output_1 = self.prediction_model(
                            proj_head_output_1, training=True)

                        # Target
                        proj_head_output_2, supervised_head_output_2 = self.target_model(
                            images_two, training=True)

                        # -------------------------------------------------------------
                        # Passing Image 1, Image 2 to Target Encoder,  Online Encoder
                        # -------------------------------------------------------------

                        # online
                        proj_head_output_2_online, _ = self.online_model(
                            images_two, training=True)
                        # Vector Representation from Online encoder go into Projection head again
                        proj_head_output_2_online = self.prediction_model(
                            proj_head_output_2_online, training=True)

                        # Target
                        proj_head_output_1_target, _ = self.target_model(
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
                            metrics.update_pretrain_metrics_train(self.metric_dict['contrast_loss_metric'],
                                                                  self.metric_dict['contrast_acc_metric'],
                                                                  self.metric_dict['contrast_entropy_metric'],
                                                                  loss, logits_ab,
                                                                  labels)
                ## 
                else: 
                    #(ds_1, ds_2, ds_3, ds_4, ds_5)
                    ## Global view :  (ds_1, ds_2)
                    # Online
                    proj_head_output_1, supervised_head_output_1 = self.online_model(
                        images_one, training=True)
                    proj_head_output_1 = self.prediction_model(
                        proj_head_output_1, training=True)
                    # Target
                    proj_head_output_2, supervised_head_output_2 = self.target_model(
                        images_two, training=True)


                    # -------------------------------------------------------------
                    # Passing Image 1, Image 2 to Target Encoder,  Online Encoder
                    # -------------------------------------------------------------

                    # online
                    proj_head_output_2_online, _ = self.online_model(
                        images_two, training=True)
                    # Vector Representation from Online encoder go into Projection head again
                    proj_head_output_2_online = self.prediction_model(
                        proj_head_output_2_online, training=True)

                    # Target
                    proj_head_output_1_target, _ = self.target_model(
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
                        metrics.update_pretrain_metrics_train(self.metric_dict['contrast_loss_metric'],
                                                              self.metric_dict['contrast_acc_metric'],
                                                              self.metric_dict['contrast_entropy_metric'],
                                                              loss, logits_ab,
                                                              labels)
            
            ## now we are in this branch!!
            elif FLAGS.loss_type == "byol_asymmetrized_loss":
                logging.info("You implement Asymmetrized loss")
                # -------------------------------------------------------------
                # Passing image 1, image 2 to Online Encoder , Target Encoder
                # -------------------------------------------------------------

                ## Global / Local view : 
                # global view 
                proj_head_output_1, supervised_head_output_1 = self.online_model(
                    ds_1, training=True)

                proj_head_output_1 = self.prediction_model(  # asym MLP
                    proj_head_output_1, training=True)

                # Target
                proj_head_output_2, supervised_head_output_2 = self.target_model(
                    ds_2, training=True)

                # Local view 
                proj_head_output_3, _ = self.online_model(
                    ds_3, training=True)
                proj_head_output_3 = self.prediction_model(
                    proj_head_output_3, training=True)

                # pair-target
                proj_head_output_34, _ = self.target_model(
                    ds_4, training=True)
                # pair-target
                proj_head_output_35, _ = self.target_model(
                    ds_5, training=True)
                

                # Compute Contrastive Train Loss -->
                loss = None
                if proj_head_output_1 is not None:
                    # Compute Contrastive Loss model
                    ## loss measurement : 
                    loss, logits_ab, labels = distributed_loss(
                        proj_head_output_1, proj_head_output_2,  proj_head_output_3, proj_head_output_34,  proj_head_output_35)

                    if loss is None:
                        loss = loss
                    else:
                        loss += loss

                    # Update Self-Supervised Metrics
                    metrics.update_pretrain_metrics_train(self.metric_dict['contrast_loss_metric'],
                                                            self.metric_dict['contrast_acc_metric'],
                                                            self.metric_dict['contrast_entropy_metric'],
                                                            loss, logits_ab,
                                                            labels)
            else:
                raise ValueError("Invalid Type loss")

            # Compute the Supervised train Loss
            '''Consider Sperate Supervised Loss'''

            # supervised_loss=None
            if supervised_head_output_1 is not None:

                if self.train_mode == 'pretrain' and self.lineareval_while_pretraining:

                    if FLAGS.XLA_compiler == "model_only":
                        with tf.xla.experimental.jit_scope():
                            outputs = tf.concat(
                                [supervised_head_output_1, supervised_head_output_2], 0)
                            supervise_lable = tf.concat(
                                [lable_one, lable_two], 0)

                            # Calculte the cross_entropy loss with Labels
                            sup_loss = obj_lib.add_supervised_loss(
                                labels=supervise_lable, logits=outputs)

                            scale_sup_loss = tf.nn.compute_average_loss(
                                sup_loss, global_batch_size=self.train_global_batch)
                            # scale_sup_loss = tf.reduce_sum(
                            #     sup_loss) * (1./train_global_batch)
                            # Update Supervised Metrics
                            metrics.update_finetune_metrics_train(self.metric_dict['supervised_loss_metric'],
                                                                  self.metric_dict['supervised_acc_metric'],
                                                                  scale_sup_loss, supervise_lable, outputs)

                    else:

                        outputs = tf.concat(
                            [supervised_head_output_1, supervised_head_output_2], 0)
                        supervise_lable = tf.concat(
                            [lable_one, lable_two], 0)

                        # Calculte the cross_entropy loss with Labels
                        sup_loss = obj_lib.add_supervised_loss(
                            labels=supervise_lable, logits=outputs)

                        scale_sup_loss = tf.nn.compute_average_loss(
                            sup_loss, global_batch_size=self.train_global_batch)
                        # scale_sup_loss = tf.reduce_sum(
                        #     sup_loss) * (1./train_global_batch)
                        # Update Supervised Metrics
                        metrics.update_finetune_metrics_train(self.metric_dict['supervised_loss_metric'],
                                                              self.metric_dict['supervised_acc_metric'],
                                                              scale_sup_loss, supervise_lable, outputs)

                '''Attention'''
                # Noted Consideration Aggregate (Supervised + Contrastive Loss) --> Update the Model Gradient
                if self.aggregate_loss == "contrastive_supervised":
                    if loss is None:
                        loss = scale_sup_loss
                    else:
                        loss += scale_sup_loss

                elif self.aggregate_loss == "contrastive":

                    supervise_loss = None
                    if supervise_loss is None:
                        supervise_loss = scale_sup_loss
                    else:
                        supervise_loss += scale_sup_loss
                else:
                    raise ValueError(
                        " Loss aggregate is invalid please check self.aggregate_loss")

            weight_decay_loss = all_model.add_weight_decay(
                self.online_model, adjust_per_optimizer=True)
            # Under experiment Scale loss after adding Regularization and scaled by Batch_size
            # weight_decay_loss = tf.nn.scale_regularization_loss(
            #     weight_decay_loss)
            self.metric_dict['weight_decay_metric'].update_state(
                weight_decay_loss)
            loss += weight_decay_loss
            self.metric_dict['total_loss_metric'].update_state(loss)

            logging.info('Trainable variables:')
            for var in self.online_model.trainable_variables:
                logging.info(var.name)

        if FLAGS.mixprecision == 'FP32':

            # Update Encoder and Projection head weight
            grads = tape.gradient(loss, self.online_model.trainable_variables)
            self.opt.apply_gradients(
                zip(grads, self.online_model.trainable_variables))

            # Update Prediction Head model
            grads = tape.gradient(
                loss, self.prediction_model.trainable_variables)
            self.opt.apply_gradients(
                zip(grads, self.prediction_model.trainable_variables))

        elif FLAGS.mixprecision == 'FP16':

            fp32_grads = tape.gradient(
                loss, self.online_model.trainable_variables)
            fp16_grads = [tf.cast(grad, 'float16') for grad in fp32_grads]
            all_reduce_fp16_grads = tf.distribute.get_replica_context(
            ).all_reduce(tf.distribute.ReduceOp.SUM, fp16_grads)
            # all_reduce_fp32_grads = [
            #     tf.cast(grad, 'float32') for grad in all_reduce_fp16_grads]
            all_reduce_fp32_grads = self.opt.get_unscaled_gradients(
                all_reduce_fp16_grads)

            self.opt.apply_gradients(zip(all_reduce_fp32_grads, self.online_model.trainable_variables),
                                     experimental_aggregate_gradients=False)

            fp32_grads = tape.gradient(
                loss, self.prediction_model.trainable_variables)
            fp16_grads = [tf.cast(grad, 'float16')for grad in fp32_grads]
            all_reduce_fp16_grads = tf.distribute.get_replica_context(
            ).all_reduce(tf.distribute.ReduceOp.SUM, fp16_grads)
            # all_reduce_fp32_grads = [
            #     tf.cast(grad, 'float32') for grad in all_reduce_fp16_grads]
            all_reduce_fp32_grads = self.opt.get_unscaled_gradients(
                all_reduce_fp16_grads)
            self.opt.apply_gradients(zip(
                all_reduce_fp32_grads, self.prediction_model.trainable_variables),
                experimental_aggregate_gradients=False)
        else:
            raise ValueError("Not supporting training Precision")

        del tape
        return loss


if __name__ == '__main__':
    from config.absl_mock import Mock_Flag
    from config.non_contrast_config_v1 import read_cfg_base

    # dummy assignment, so let it in one line
    flag = read_cfg_base()
    FLAGS = flag.FLAGS

    if not os.path.isdir(FLAGS.model_dir):
        print("Creat the model dir: ", FLAGS.model_dir)
        os.makedirs(FLAGS.model_dir)

    set_gpu_env()

    wanda_cfg = {
        "Model_Arch": f"ResNet{FLAGS.resnet_depth}",
        "Training mode": "Baseline Non_Contrastive",
        "DataAugmentation_types": "Inception_Croping_Auto_Augment",
        "Dataset": "ImageNet1k",

        "IMG_SIZE": FLAGS.image_size,
        "Epochs": FLAGS.train_epochs,
        "Batch_size": None,   # this will be fill in during the run-time
        "Learning_rate": FLAGS.base_lr,
        "Temperature": FLAGS.temperature,
        "Optimizer": FLAGS.optimizer,
        "SEED": FLAGS.SEED,
        "Subset_dataset": FLAGS.num_classes,
        "Loss type": FLAGS.aggregate_loss,
        "opt": FLAGS.up_scale
    }
    runer = Runner(FLAGS, wanda_cfg)
    if "train" in FLAGS.mode:
        runer.train(FLAGS.mode)
    else:  # perform evaluation
        runer.eval()