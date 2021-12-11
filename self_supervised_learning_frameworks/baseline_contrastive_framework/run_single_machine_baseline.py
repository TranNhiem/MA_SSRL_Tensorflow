import os
import json
import math
import wandb
import random
from absl import flags
from absl import logging
from absl import app

import tensorflow as tf
from learning_rate_optimizer import WarmUpAndCosineDecay
import metrics
from byol_simclr_imagenet_data_harry import imagenet_dataset_single_machine
from self_supervised_losses import nt_xent_symetrize_loss_simcrl, nt_xent_asymetrize_loss_v2
import model as all_model
import objective as obj_lib
from imutils import paths

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

FLAGS = flags.FLAGS
#------------------------------------------
# General Define
#------------------------------------------

flags.DEFINE_integer(
    'IMG_height', 224,
    'image height.')

flags.DEFINE_integer(
    'IMG_width', 224,
    'image width.')

flags.DEFINE_float(
    'LARGE_NUM', 1e9,
    'LARGE_NUM to multiply with Logit.')

flags.DEFINE_integer(
    'image_size', 224,
    'image size.')

flags.DEFINE_integer(
    'num_classes', 1000,
    'Number of class in dataset.'
)

flags.DEFINE_integer(
    'SEED', 26,
    'random seed use for shuffle data Generate two same image ds_one & ds_two')

flags.DEFINE_integer(
    'SEED_data_split', 100,
    'random seed for spliting data the same for all the run with the same validation dataset.')

flags.DEFINE_integer(
    'train_batch_size', 128,
    'Train batch_size .')

flags.DEFINE_integer(
    'val_batch_size', 128,
    'Validaion_Batch_size.')

flags.DEFINE_integer(
    'train_epochs', 50,
    'Number of epochs to train for.')

flags.DEFINE_string(
   
    'train_path', '/data1/1K_New/train/',
    'Train dataset path.')

flags.DEFINE_string(
  
    'val_path', "/data1/1K_New/val/",
    'Validaion dataset path.')


## Mask_folder should locate in location and same level of train folder
flags.DEFINE_string(
    'mask_path', "train_binary_mask_by_USS",
    'Mask path.')

flags.DEFINE_string(
    'train_label', "image_net_1k_lable.txt",
    'train_label.')

flags.DEFINE_string(
    'val_label', "ILSVRC2012_validation_ground_truth.txt",
    'val_label.')

#------------------------------------------
# Define for Linear Evaluation
#------------------------------------------
flags.DEFINE_enum(
    'linear_evaluate', 'standard', ['standard', 'randaug', 'cropping_randaug'],
    'How to scale the learning rate as a function of batch size.')

flags.DEFINE_integer(
    'eval_steps', 0,
    'Number of steps to eval for. If not provided, evals over entire dataset.')
# Configure RandAugment for validation dataset augmentation transform

flags.DEFINE_float(
    'randaug_transform', 1,
    'Number of augmentation transformations.')

flags.DEFINE_float(
    'randaug_magnitude', 7,
    'Number of augmentation transformations.')

#----------------------------------------------------------
# Define for Learning Rate Optimizer + Training Strategy
#----------------------------------------------------------

# Learning Rate Scheudle

flags.DEFINE_float(
    'base_lr', 0.3,
    'Initial learning rate per batch size of 256.')

flags.DEFINE_integer(
    'warmup_epochs', 10,  # Configure BYOL and SimCLR
    'warmup epoch steps for Cosine Decay learning rate schedule.')


flags.DEFINE_enum(
    'lr_rate_scaling', 'linear', ['linear', 'sqrt', 'no_scale', ],
    'How to scale the learning rate as a function of batch size.')

# Optimizer
# Same the Original SimClRV2 training Configure
'''ATTENTION'''
flags.DEFINE_enum(

    # if Change the Optimizer please change --
    'optimizer', 'LARSW', ['Adam', 'SGD', 'LARS', 'AdamW', 'SGDW', 'LARSW',
                          'AdamGC', 'SGDGC', 'LARSGC', 'AdamW_GC', 'SGDW_GC', 'LARSW_GC'],
    'How to scale the learning rate as a function of batch size.')

flags.DEFINE_enum(
    # Same the Original SimClRV2 training Configure
    # 1. original for ['Adam', 'SGD', 'LARS']
    # 2.optimizer_weight_decay for ['AdamW', 'SGDW', 'LARSW']
    # 3. optimizer_GD fir  ['AdamGC', 'SGDGC', 'LARSGC']
    # 4. optimizer_W_GD for ['AdamW_GC', 'SGDW_GC', 'LARSW_GC']

    'optimizer_type', 'optimizer_weight_decay', ['original', 'optimizer_weight_decay','optimizer_GD','optimizer_W_GD' ],
    'Optimizer type corresponding to Configure of optimizer')

flags.DEFINE_float(
    'momentum', 0.9,
    'Momentum parameter.')

flags.DEFINE_float('weight_decay', 1e-6, 'Amount of weight decay to use.')

#----------------------------------------------------------------------
# Configure for Encoder - Projection Head, Linear Evaluation Architecture
#----------------------------------------------------------------------

# Encoder Configure

flags.DEFINE_boolean(
    'global_bn', True,
    'Whether to aggregate BN statistics across distributed cores.')

flags.DEFINE_float(
    'batch_norm_decay', 0.9,  # Checkout BN decay concept
    'Batch norm decay parameter.')

flags.DEFINE_integer(
    'width_multiplier', 1,
    'Multiplier to change width of network.')

flags.DEFINE_integer(
    'resnet_depth', 50,
    'Depth of ResNet.')

flags.DEFINE_float(
    'sk_ratio', 0.,
    'If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')

flags.DEFINE_float(
    'se_ratio', 0.,
    'If it is bigger than 0, it will enable SE.')

# Projection Head

flags.DEFINE_enum(
    'proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],
    'How the head projection is done.')

flags.DEFINE_integer(
    'proj_out_dim', 128,
    'Number of head projection dimension.')

flags.DEFINE_integer(
    'num_proj_layers', 3,
    'Number of non-linear head layers.')

flags.DEFINE_integer(
    'ft_proj_selector', 0,
    'Which layer of the projection head to use during fine-tuning. '
    '0 means no projection head, and -1 means the final layer.')

flags.DEFINE_float(
    'temperature', 0.5,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_boolean(
    'hidden_norm', True,
    'L2 Normalization Vector representation.')

#-----------------------------------------
# Configure Model Training
#-----------------------------------------

# Self-Supervised training and Supervised training mode
flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'train_then_eval'],
    'Whether to perform training or evaluation.')

flags.DEFINE_enum(
    'train_mode', 'pretrain', ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.')

flags.DEFINE_bool('lineareval_while_pretraining', True,
                  'Whether to finetune supervised head while pretraining.')

flags.DEFINE_enum(
    'aggregate_loss', 'contrastive_supervised', [
        'contrastive', 'contrastive_supervised', ],
    'Consideration update Model with One Contrastive or sum up and (Contrastive + Supervised Loss).')

flags.DEFINE_enum(
    'loss_options' , 'loss_v1', 
    ['loss_v0', 'loss_v1'], 
    "Option for chossing loss version [V0]--> Original simclr loss [V1] --> Custom build design loss"
)




# Fine Tuning configure

flags.DEFINE_bool(
    'zero_init_logits_layer', False,
    'If True, zero initialize layers after avg_pool for supervised learning.')

flags.DEFINE_integer(
    'fine_tune_after_block', -1,
    'The layers after which block that we will fine-tune. -1 means fine-tuning '
    'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
    'just the linear head.')

#-----------------------------------------
# Configure Saving and Restore Model
#-----------------------------------------

# Saving Model

flags.DEFINE_string(
    'model_dir', "./model_ckpt/simclrResNet/",
    'Model directory for training.')

flags.DEFINE_integer(
    'keep_hub_module_max', 1,
    'Maximum number of Hub modules to keep.')

flags.DEFINE_integer(
    'keep_checkpoint_max', 5,
    'Maximum number of checkpoints to keep.')


# Loading Model

# Restore model weights only, but not global step and optimizer states
flags.DEFINE_string(
    'checkpoint', None,
    'Loading from the given checkpoint for fine-tuning if a finetuning '
    'checkpoint does not already exist in model_dir.')

flags.DEFINE_integer(
    'checkpoint_epochs', 1,
    'Number of epochs between checkpoints/summaries.')

flags.DEFINE_integer(
    'checkpoint_steps', 100,
    'Number of steps between checkpoints/summaries. If provided, overrides '
    'checkpoint_epochs.')

#-------------------------------------------------------------
# Helper function to save and resore model.
#-------------------------------------------------------------

def get_salient_tensors_dict(include_projection_head):
    """Returns a dictionary of tensors."""
    graph = tf.compat.v1.get_default_graph()
    result = {}
    for i in range(1, 5):
        result['block_group%d' % i] = graph.get_tensor_by_name(
            'resnet/block_group%d/block_group%d:0' % (i, i))
        result['initial_conv'] = graph.get_tensor_by_name(
            'resnet/initial_conv/Identity:0')
        result['initial_max_pool'] = graph.get_tensor_by_name(
            'resnet/initial_max_pool/Identity:0')
        result['final_avg_pool'] = graph.get_tensor_by_name(
            'resnet/final_avg_pool:0')

        result['logits_sup'] = graph.get_tensor_by_name(
            'head_supervised/logits_sup:0')

    if include_projection_head:
        result['proj_head_input'] = graph.get_tensor_by_name(
            'projection_head/proj_head_input:0')
        result['proj_head_output'] = graph.get_tensor_by_name(
            'projection_head/proj_head_output:0')
    return result

def build_saved_model(model, include_projection_head=True):
    """Returns a tf.Module for saving to SavedModel."""

    class SimCLRModel(tf.Module):
        """Saved model for exporting to hub."""

        def __init__(self, model):
            self.model = model
            # This can't be called `trainable_variables` because `tf.Module` has
            # a getter with the same name.
            self.trainable_variables_list = model.trainable_variables

        @tf.function
        def __call__(self, inputs, trainable):
            self.model(inputs, training=trainable)
            return get_salient_tensors_dict(include_projection_head)

    module = SimCLRModel(model)
    input_spec = tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)
    module.__call__.get_concrete_function(input_spec, trainable=True)
    module.__call__.get_concrete_function(input_spec, trainable=False)

    return module

# configure Json format saving file

def json_serializable(val):
    #
    try:
        json.dumps(val)
        return True

    except TypeError:
        return False

def save(model, global_step):
    """Export as SavedModel for finetuning and inference."""
    saved_model = build_saved_model(model)
    export_dir = os.path.join(FLAGS.model_dir, 'saved_model')
    checkpoint_export_dir = os.path.join(export_dir, str(global_step))

    if tf.io.gfile.exists(checkpoint_export_dir):
        tf.io.gfile.rmtree(checkpoint_export_dir)
    tf.saved_model.save(saved_model, checkpoint_export_dir)

    if FLAGS.keep_hub_module_max > 0:
        # Delete old exported SavedModels.
        exported_steps = []
        for subdir in tf.io.gfile.listdir(export_dir):
            if not subdir.isdigit():
                continue
            exported_steps.append(int(subdir))
        exported_steps.sort()
        for step_to_delete in exported_steps[:-FLAGS.keep_hub_module_max]:
            tf.io.gfile.rmtree(os.path.join(export_dir, str(step_to_delete)))

# Restore the checkpoint forom the file

def try_restore_from_checkpoint(model, global_step, optimizer):
    """Restores the latest ckpt if it exists, otherwise check FLAGS.checkpoint."""
    checkpoint = tf.train.Checkpoint(
        model=model, global_step=global_step, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=FLAGS.model_dir,
        max_to_keep=FLAGS.keep_checkpoint_max)
    latest_ckpt = checkpoint_manager.latest_checkpoint

    if latest_ckpt:
        # Restore model weights, global step, optimizer states
        logging.info('Restoring from latest checkpoint: %s', latest_ckpt)
        checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()

    elif FLAGS.checkpoint:
        # Restore model weights only, but not global step and optimizer states
        logging.info('Restoring from given checkpoint: %s', FLAGS.checkpoint)
        checkpoint_manager2 = tf.train.CheckpointManager(
            tf.train.Checkpoint(model=model),
            directory=FLAGS.model_dir,
            max_to_keep=FLAGS.keep_checkpoint_max)
        checkpoint_manager2.checkpoint.restore(
            FLAGS.checkpoint).expect_partial()

    if FLAGS.zero_init_logits_layer:
        model = checkpoint_manager2.checkpoint.model
        output_layer_parameters = model.supervised_head.trainable_weights
        logging.info('Initializing output layer parameters %s to zero',
                     [x.op.name for x in output_layer_parameters])
        for x in output_layer_parameters:
            x.assign(tf.zeros_like(x))

    return checkpoint_manager

def _restore_latest_or_from_pretrain(checkpoint_manager):
    """Restores the latest ckpt if training already.
    Or restores from FLAGS.checkpoint if in finetune mode.
    Args:
    checkpoint_manager: tf.traiin.CheckpointManager.
    """
    latest_ckpt = checkpoint_manager.latest_checkpoint

    if latest_ckpt:
        # The model is not build yet so some variables may not be available in
        # the object graph. Those are lazily initialized. To suppress the warning
        # in that case we specify `expect_partial`.
        logging.info('Restoring from %s', latest_ckpt)
        checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()

    elif FLAGS.train_mode == 'finetune':
        # Restore from pretrain checkpoint.
        assert FLAGS.checkpoint, 'Missing pretrain checkpoint.'
        logging.info('Restoring from %s', FLAGS.checkpoint)
        checkpoint_manager.checkpoint.restore(
            FLAGS.checkpoint).expect_partial()
        # TODO(iamtingchen): Can we instead use a zeros initializer for the
        # supervised head?

    if FLAGS.zero_init_logits_layer:
        model = checkpoint_manager.checkpoint.model
        output_layer_parameters = model.supervised_head.trainable_weights
        logging.info('Initializing output layer parameters %s to zero',
                     [x.op.name for x in output_layer_parameters])

        for x in output_layer_parameters:
            x.assign(tf.zeros_like(x))

# Perform Testing Step Here
def perform_evaluation(model, val_ds, val_steps, ckpt, strategy):
    """Perform evaluation.--> Only Inference to measure the pretrain model representation"""

    if FLAGS.train_mode == 'pretrain' and not FLAGS.lineareval_while_pretraining:
        logging.info('Skipping eval during pretraining without linear eval.')
        return

    # Tensorboard enable
    summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)

    # Building the Supervised metrics
    with strategy.scope():

        regularization_loss = tf.keras.metrics.Mean('eval/regularization_loss')
        label_top_1_accuracy = tf.keras.metrics.Accuracy(
            "eval/label_top_1_accuracy")
        label_top_5_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(
            5, 'eval/label_top_5_accuracy')

        all_metrics = [
            regularization_loss, label_top_1_accuracy, label_top_5_accuracy
        ]

        # Restore model checkpoint
        logging.info('Restoring from %s', ckpt)
        checkpoint = tf.train.Checkpoint(
            model=model, global_step=tf.Variable(0, dtype=tf.int64))
        checkpoint.restore(ckpt).expect_partial()
        global_step = checkpoint.global_step
        logging.info('Performing eval at step %d', global_step.numpy())

    # Scaling the loss  -- Update the sum up all the gradient
    @tf.function
    def single_step(features, labels):
        # Logits output
        _, supervised_head_outputs = model(features, training=False)
        assert supervised_head_outputs is not None
        outputs = supervised_head_outputs

        metrics.update_finetune_metrics_eval(
            label_top_1_accuracy, label_top_5_accuracy, outputs, labels)

        # Single machine loss
        reg_loss = all_model.add_weight_decay(model, adjust_per_optimizer=True)
        regularization_loss.update_state(reg_loss)

    with strategy.scope():

        @tf.function
        def run_single_step(iterator):
            images, labels = next(iterator)
            strategy.run(single_step, (images, labels))

    iterator = iter(val_ds)
    for i in range(val_steps):
        run_single_step(iterator)
        logging.info("Complete validation for %d step ", i+1, val_steps)

    # At this step of training with Ckpt Complete evaluate model performance
    logging.info('Finished eval for %s', ckpt)

    # Logging to tensorboard for the information
    # Write summaries
    cur_step = global_step.numpy()
    logging.info('Writing summaries for %d step', cur_step)

    with summary_writer.as_default():
        metrics.log_and_write_metrics_to_summary(all_metrics, cur_step)
        summary_writer.flush()

    # Record results as Json.
    result_json_path = os.path.join(FLAGS.model_dir, 'result.jsoin')
    result = {metric.name: metric.result().numpy() for metric in all_metrics}
    result['global_step'] = global_step.numpy()
    logging.info(result)

    with tf.io.gfile.GFile(result_json_path, 'w') as f:
        json.dump({k: float(v) for k, v in result.items()}, f)
    result_json_path = os.path.join(
        FLAGS.model_dir, 'result_%d.json' % result['global_step'])

    with tf.io.gfile.GFile(result_json_path, 'w') as f:
        json.dump({k: float(v) for k, v in result.items()}, f)
    flag_json_path = os.path.join(FLAGS.model_dir, 'flags.json')

    with tf.io.gfile.GFile(flag_json_path, 'w') as f:
        serializable_flags = {}
        for key, val in FLAGS.flag_values_dict().items():
            # Some flag value types e.g. datetime.timedelta are not json serializable,
            # filter those out.
            if json_serializable(val):
                serializable_flags[key] = val
            json.dump(serializable_flags, f)

    # Export as SavedModel for finetuning and inference.
    save(model, global_step=result['global_step'])

    return result


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Preparing dataset
    # Imagenet path prepare localy

    strategy = tf.distribute.MirroredStrategy()
    train_global_batch = FLAGS.train_batch_size * strategy.num_replicas_in_sync
    val_global_batch = FLAGS.val_batch_size * strategy.num_replicas_in_sync

    train_dataset = imagenet_dataset_single_machine(img_size=FLAGS.image_size, train_batch=train_global_batch,  val_batch=val_global_batch,
                                                    strategy=strategy, train_path=FLAGS.train_path, val_path=FLAGS.val_path,
                                                    train_label=FLAGS.train_label,val_label=FLAGS.val_label,
                                                    mask_path=FLAGS.mask_path, bi_mask=False)

    train_ds = train_dataset.simclr_random_global_crop()

    val_ds = train_dataset.supervised_validation()

    num_train_examples,num_eval_examples = train_dataset.get_data_size()

    train_steps = FLAGS.eval_steps or int(num_train_examples * FLAGS.train_epochs // train_global_batch)*2
    eval_steps = FLAGS.eval_steps or int(math.ceil(num_eval_examples / val_global_batch))

    epoch_steps = int(round(num_train_examples / train_global_batch))
    checkpoint_steps = (FLAGS.checkpoint_steps or ( FLAGS.checkpoint_epochs * epoch_steps))

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
        "Loss type":FLAGS.aggregate_loss,
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
                
                if FLAGS.loss_options =="loss_v0": 
                    # each GPU loss per_replica batch loss
                    per_example_loss, logits_ab, labels = nt_xent_symetrize_loss_simcrl(
                        x1, x2, LARGE_NUM=FLAGS.LARGE_NUM, hidden_norm=FLAGS.hidden_norm, temperature=FLAGS.temperature)
                    
                elif FLAGS.loss_options =="loss_v1": 
                    # each GPU loss per_replica batch loss
                    x_1_2= tf.concat([x1, x2], axis=0)
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
                        loss, logits_ab, labels = distributed_loss( proj_head_output_1, proj_head_output_2)
                        
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
                            
                            outputs = tf.concat([supervised_head_output_1, supervised_head_output_2], 0)
                            supervise_lable = tf.concat([lable_one, lable_two], 0)

                            # Calculte the cross_entropy loss with Labels
                            sup_loss = obj_lib.add_supervised_loss(labels=supervise_lable, logits=outputs)
                            
                            scale_sup_loss = tf.nn.compute_average_loss(sup_loss, global_batch_size=train_global_batch)
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
                        if FLAGS.aggregate_loss== "contrastive_supervised": 
                            
                            if loss is None:
                                loss = scale_sup_loss
                            else:
                                loss += scale_sup_loss

                        elif FLAGS.aggregate_loss== "contrastive":
                           
                            supervise_loss=None
                            if supervise_loss is None:
                                supervise_loss = scale_sup_loss
                            else:
                                supervise_loss += scale_sup_loss
                        else: 
                            raise ValueError(" Loss aggregate is invalid please check FLAGS.aggregate_loss")
                    
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
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

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
                num_batches=0
                for _, (ds_one, ds_two) in enumerate(train_ds):

                    total_loss += distributed_train_step(ds_one, ds_two)
                    num_batches+=1
                    #if (global_step.numpy()+ 1) % checkpoint_steps==0:
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

                epoch_loss= total_loss/num_batches
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
