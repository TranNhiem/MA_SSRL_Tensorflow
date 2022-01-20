import os
import json
import math
import random
import tensorflow as tf
from absl import logging
from Neural_Net_Architecture.Convolution_Archs.ResNet_models import ssl_model as all_model
from .visualize import Visualize
# -------------------------------------------------------------
# Helper function to save and resore model.
# -------------------------------------------------------------

#FLAGS= flags.FLAGS
import objectives.metrics as metrics
from config.absl_mock import Mock_Flag
flag = Mock_Flag()
FLAGS = flag.FLAGS


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

    if FLAGS.restore_checkpoint:
        if latest_ckpt:
            # Restore model weights, global step, optimizer states
            logging.info('Restoring from latest checkpoint: %s', latest_ckpt)
            checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()

    elif FLAGS.checkpoint:
        print("in")
        # Restore model weights only, but not global step and optimizer states
        logging.info('Restoring from given checkpoint: %s',
                     FLAGS.checkpoint)
        checkpoint_manager2 = tf.train.CheckpointManager(
            tf.train.Checkpoint(model=model),
            directory=FLAGS.model_dir,
            max_to_keep=FLAGS.keep_checkpoint_max)
        checkpoint_manager2.checkpoint.restore(
            FLAGS.checkpoint).expect_partial()
    else:
        logging.info('You are Not Restore from Checkpoint: %s', latest_ckpt)

    if FLAGS.zero_init_logits_layer:
        print("in2")
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
        print("Successful loading checkpt")
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

    # # Tensorboard enable
    # summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)

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
        print("start v")
        _, _, _, supervised_head_outputs = model(features, training=False)
        v = Visualize(1, FLAGS.visualize_dir)
        v.plot_feature_map(1, supervised_head_outputs)
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

    # with summary_writer.as_default():
    #     metrics.log_and_write_metrics_to_summary(all_metrics, cur_step)
    #     summary_writer.flush()

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

    # with tf.io.gfile.GFile(flag_json_path, 'w') as f:
    #     serializable_flags = {}
    #     for key, val in FLAGS.flag_values_dict().items():
    #         # Some flag value types e.g. datetime.timedelta are not json serializable,
    #         # filter those out.
    #         if json_serializable(val):
    #             serializable_flags[key] = val
    #         json.dump(serializable_flags, f)

    # # Export as SavedModel for finetuning and inference.
    # save(model, global_step=result['global_step'])

    return result
