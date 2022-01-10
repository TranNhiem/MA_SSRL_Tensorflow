# coding=utf-8
# Inherence model design from SimCLR Authors.
# ==============================================================================

import math
from losses_optimizers import lars_optimizer
from . import resnet
from .resnet_modify import resnet as resnet_modify
import tensorflow as tf
from losses_optimizers.learning_rate_optimizer import get_optimizer
from tensorflow.keras import mixed_precision
from config.absl_mock import Mock_Flag

flag = Mock_Flag()
FLAGS = flag.FLAGS

# Equivalent to the two lines above

# if FLAGS.mixprecision == "fp16":
#     mixed_precision.set_global_policy('mixed_float16')


def build_optimizer(lr_schedule):
    '''
    Args
    lr_schedule: learning values.

    Return:
    'original', 'optimizer_weight_decay','optimizer_GD','optimizer_W_GD' 
    optimizer.
    '''
    if FLAGS.optimizer_type == "original":
        Optimizer_type = FLAGS.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        optimizer = optimizers.original_optimizer(FLAGS)

    elif FLAGS.optimizer_type == "optimizer_weight_decay":
        Optimizer_type = FLAGS.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        optimizer = optimizers.optimizer_weight_decay(FLAGS)

    elif FLAGS.optimizer_type == "optimizer_GD":
        Optimizer_type = FLAGS.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        optimizer = optimizers.optimizer_gradient_centralization(FLAGS)

    elif FLAGS.optimizer_type == "optimizer_W_GD":
        Optimizer_type = FLAGS.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        optimizer = optimizers.optimizer_weight_decay_gradient_centralization(
            FLAGS)
    else:
        raise ValueError(" FLAGS.Optimizer type is invalid please check again")
    #optimizer_mix_percision = mixed_precision.LossScaleOptimizer(optimizer)

    if FLAGS.mixprecision == "fp16":
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    return optimizer


def build_optimizer_multi_machine(lr_schedule):
    '''
    Args
    lr_schedule: learning values.

    Return:
    The mix_percision optimizer.'optimizer_weight_decay','optimizer_GD','optimizer_W_GD' 
    '''

    if FLAGS.optimizer_type == "original":
        Optimizer_type = FLAGS.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        optimizer = optimizers.original_optimizer(FLAGS)
        optimizer_mix_percision = mixed_precision.LossScaleOptimizer(optimizer)
    elif FLAGS.optimizer_type == "optimizer_weight_decay":
        Optimizer_type = FLAGS.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        optimizer = optimizers.optimizer_weight_decay(FLAGS)
        optimizer_mix_percision = mixed_precision.LossScaleOptimizer(optimizer)

    elif FLAGS.optimizer_type == "optimizer_GD":
        Optimizer_type = FLAGS.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        optimizer = optimizers.optimizer_gradient_centralization(FLAGS)
        optimizer_mix_percision = mixed_precision.LossScaleOptimizer(optimizer)

    elif FLAGS.optimizer_type == "optimizer_W_GD":
        Optimizer_type = FLAGS.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        optimizer = optimizers.optimizer_weight_decay_gradient_centralization(
            FLAGS)
        optimizer_mix_percision = mixed_precision.LossScaleOptimizer(optimizer)
    else:
        raise ValueError(" FLAGS.Optimizer type is invalid please check again")
    #optimizer_mix_percision = mixed_precision.LossScaleOptimizer(optimizer)

    return optimizer_mix_percision


def add_weight_decay(model, adjust_per_optimizer=True):
    """Compute weight decay from flags."""
    if adjust_per_optimizer and 'lars' in FLAGS.optimizer:
        # Weight decay are taking care of by optimizer for these cases.
        # Except for supervised head, which will be added here.
        #
        l2_losses = [
            tf.nn.l2_loss(v)
            for v in model.trainable_variables
            if 'head_supervised' in v.name and 'bias' not in v.name
        ]
        if l2_losses:
            return FLAGS.weight_decay * tf.add_n(l2_losses)
        else:
            return 0

    # TODO(srbs): Think of a way to avoid name-based filtering here.
    l2_losses = [
        tf.nn.l2_loss(v)
        for v in model.trainable_weights
        if 'batch_normalization' not in v.name
    ]

    loss = FLAGS.weight_decay * tf.add_n(l2_losses)

    return loss


class LinearLayer(tf.keras.layers.Layer):

    def __init__(self,
                 num_classes,
                 use_bias=True,
                 use_bn=False,
                 name='linear_layer',
                 **kwargs):
        # Note: use_bias is ignored for the dense layer when use_bn=True.
        # However, it is still used for batch norm.
        super(LinearLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.use_bias = use_bias
        self.use_bn = use_bn
        self._name = name
        if self.use_bn:
            self.bn_relu = resnet.BatchNormRelu(relu=False, center=use_bias)

    def build(self, input_shape):
        # TODO(srbs): Add a new SquareDense layer.
        if callable(self.num_classes):
            num_classes = self.num_classes(input_shape)
        else:
            num_classes = self.num_classes
        self.dense = tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            use_bias=self.use_bias and not self.use_bn)

        super(LinearLayer, self).build(input_shape)

    def call(self, inputs, training):
        assert inputs.shape.ndims == 2, inputs.shape
        inputs = self.dense(inputs)
        if self.use_bn:
            inputs = self.bn_relu(inputs, training=training)
        return inputs

# Linear Layers tf.keras.layer.Dense


class modify_LinearLayer(tf.keras.layers.Layer):

    def __init__(self,
                 num_classes,
                 up_scale=4096,
                 non_contrastive=False,
                 use_bias=True,
                 use_bn=False,
                 name='linear_layer',
                 **kwargs):
        # Note: use_bias is ignored for the dense layer when use_bn=True.
        # However, it is still used for batch norm.
        super(modify_LinearLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.up_scale = up_scale
        self.use_bias = use_bias
        self.use_bn = use_bn
        self._name = name
        self.non_contrastive = non_contrastive
        if self.use_bn:
            self.bn_relu = resnet.BatchNormRelu(relu=False, center=use_bias)
            #self.bn_relu= tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        # TODO(srbs): Add a new SquareDense layer.
        if callable(self.num_classes):
            num_classes = self.num_classes(input_shape)

        else:
            num_classes = self.num_classes

        self.dense_upscale = tf.keras.layers.Dense(
            self.up_scale,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            use_bias=self.use_bias and not self.use_bn, )

        self.dense = tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            use_bias=self.use_bias and not self.use_bn)

        super(modify_LinearLayer, self).build(input_shape)

    def call(self, inputs, training):
        assert inputs.shape.ndims == 2, inputs.shape
        if self.non_contrastive:
            inputs = self.dense_upscale(inputs)
            # print(inputs.shape)
            #inputs = self.dense(inputs)
            if self.use_bn:
                inputs = self.bn_relu(inputs, training=training)
        else:
            inputs = self.dense(inputs)
            # print(inputs.shape)
            if self.use_bn:
                inputs = self.bn_relu(inputs, training=training)
        return inputs

# 1 Dense Linear Classify model


class SupervisedHead(tf.keras.layers.Layer):

    def __init__(self, num_classes, name='head_supervised', **kwargs):
        super(SupervisedHead, self).__init__(name=name, **kwargs)
        self.linear_layer = modify_LinearLayer(num_classes)

    def call(self, inputs, training):
        inputs = self.linear_layer(inputs, training)
        inputs = tf.identity(inputs, name='logits_sup')
        return inputs


class ProjectionHead(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        out_dim = FLAGS.proj_out_dim
        self.linear_layers = []
        if FLAGS.proj_head_mode == 'none':
            pass  # directly use the output hiddens as hiddens
        elif FLAGS.proj_head_mode == 'linear':

            self.linear_layers = [
                LinearLayer(
                    num_classes=out_dim, use_bias=False, use_bn=True, name='l_0')
            ]
        elif FLAGS.proj_head_mode == 'nonlinear':

            for j in range(FLAGS.num_proj_layers):
                if j != FLAGS.num_proj_layers - 1:
                    # for the middle layers, use bias and relu for the output.
                    self.linear_layers.append(
                        LinearLayer(
                            num_classes=lambda input_shape: int(
                                input_shape[-1]),
                            use_bias=True,
                            use_bn=True,
                            name='nl_%d' % j))
                else:
                    # for the final layer, neither bias nor relu is used.
                    self.linear_layers.append(
                        LinearLayer(
                            num_classes=FLAGS.proj_out_dim,
                            use_bias=False,
                            use_bn=True,
                            name='nl_%d' % j))
        else:
            raise ValueError('Unknown head projection mode {}'.format(
                FLAGS.proj_head_mode))

        super(ProjectionHead, self).__init__(**kwargs)

    def call(self, inputs, training):
        if FLAGS.proj_head_mode == 'none':
            return inputs  # directly use the output hiddens as hiddens
        hiddens_list = [tf.identity(inputs, 'proj_head_input')]

        if FLAGS.proj_head_mode == 'linear':
            assert len(self.linear_layers) == 1, len(self.linear_layers)
            return hiddens_list.append(self.linear_layers[0](hiddens_list[-1],
                                                             training))
        elif FLAGS.proj_head_mode == 'nonlinear':
            for j in range(FLAGS.num_proj_layers):
                hiddens = self.linear_layers[j](hiddens_list[-1], training)
                if j != FLAGS.num_proj_layers - 1:
                    # for the middle layers, use bias and relu for the output.
                    hiddens = tf.nn.relu(hiddens)
                hiddens_list.append(hiddens)

        else:
            raise ValueError('Unknown head projection mode {}'.format(
                FLAGS.proj_head_mode))

        # The first element is the output of the projection head.
        # The second element is the input of the finetune head.
        proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')

        return proj_head_output, hiddens_list[FLAGS.ft_proj_selector]


class ProjectionHead_modify(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        out_dim = FLAGS.proj_out_dim
        self.linear_layers = []
        if FLAGS.proj_head_mode == 'none':
            pass  # directly use the output hiddens as hiddens
        elif FLAGS.proj_head_mode == 'linear':
            self.linear_layers = [
                modify_LinearLayer(
                    num_classes=out_dim,  use_bias=False, use_bn=True, name='l_0')
            ]
        elif FLAGS.proj_head_mode == 'nonlinear':
            if FLAGS.num_proj_layers > 2:
                for j in range(FLAGS.num_proj_layers):
                    if j == 0:
                        self.linear_layers.append(
                            modify_LinearLayer(
                                num_classes=lambda input_shape: int(
                                    input_shape[-1]),
                                up_scale=FLAGS.up_scale, non_contrastive=FLAGS.non_contrastive,
                                use_bias=True,
                                use_bn=True,
                                name='nl_%d' % j))

                    elif j != FLAGS.num_proj_layers - 1:
                        # for the middle layers, use bias and relu for the output.
                        if FLAGS.reduce_linear_dimention:
                            print("You Implement reduction")
                            self.linear_layers.append(
                                modify_LinearLayer(
                                    num_classes=lambda input_shape: int(
                                        input_shape[-1]/2),
                                    up_scale=FLAGS.up_scale, non_contrastive=False,
                                    use_bias=True,
                                    use_bn=True,
                                    name='nl_%d' % j))
                        else:
                            self.linear_layers.append(
                                modify_LinearLayer(
                                    num_classes=lambda input_shape: int(
                                        input_shape[-1]),
                                    up_scale=FLAGS.up_scale, non_contrastive=False,
                                    use_bias=True,
                                    use_bn=True,
                                    name='nl_%d' % j))

                    else:
                        # for the final layer, neither bias nor relu is used.
                        self.linear_layers.append(
                            modify_LinearLayer(
                                num_classes=FLAGS.proj_out_dim,
                                use_bias=False,
                                use_bn=True,
                                name='nl_%d' % j))

            else:
                for j in range(FLAGS.num_proj_layers):
                    if j != FLAGS.num_proj_layers - 1:
                        # for the middle layers, use bias and relu for the output.
                        self.linear_layers.append(
                            modify_LinearLayer(
                                num_classes=lambda input_shape: int(
                                    input_shape[-1]),
                                up_scale=FLAGS.up_scale, non_contrastive=FLAGS.non_contrastive,
                                use_bias=True,
                                use_bn=True,
                                name='nl_%d' % j))
                    else:
                        # for the final layer, neither bias nor relu is used.
                        self.linear_layers.append(
                            modify_LinearLayer(
                                num_classes=FLAGS.proj_out_dim,
                                up_scale=FLAGS.up_scale, non_contrastive=False,
                                use_bias=False,
                                use_bn=True,
                                name='nl_%d' % j))
        else:
            raise ValueError('Unknown head projection mode {}'.format(
                FLAGS.proj_head_mode))

        super(ProjectionHead_modify, self).__init__(**kwargs)

    def call(self, inputs, training):
        if FLAGS.proj_head_mode == 'none':
            return inputs  # directly use the output hiddens as hiddens
        hiddens_list = [tf.identity(inputs, 'proj_head_input')]
        if FLAGS.proj_head_mode == 'linear':
            assert len(self.linear_layers) == 1, len(self.linear_layers)
            return hiddens_list.append(self.linear_layers[0](hiddens_list[-1],
                                                             training))

        elif FLAGS.proj_head_mode == 'nonlinear':
            for j in range(FLAGS.num_proj_layers):
                hiddens = self.linear_layers[j](hiddens_list[-1], training)
                if j != FLAGS.num_proj_layers - 1:
                    # for the middle layers, use bias and relu for the output.
                    hiddens = tf.nn.relu(hiddens)
                hiddens_list.append(hiddens)

        else:
            raise ValueError('Unknown head projection mode {}'.format(
                FLAGS.proj_head_mode))

        # The first element is the output of the projection head.
        # The second element is the input of the finetune head.
        proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')

        return proj_head_output, hiddens_list[FLAGS.ft_proj_selector]


class prediction_head_model(tf.keras.models.Model):
    def __init__(self, **kwargs):

        super(prediction_head_model, self).__init__(**kwargs)
        # prediction head
        self._prediction_head = PredictionHead()

    def __call__(self, inputs, training):
        prediction_head_outputs = self._prediction_head(inputs, training)
        return prediction_head_outputs


class PredictionHead(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        out_dim = FLAGS.prediction_out_dim
        self.linear_layers = []
        if FLAGS.proj_head_mode == 'none':
            pass  # directly use the output hiddens as hiddens
        elif FLAGS.proj_head_mode == 'linear':
            self.linear_layers = [
                modify_LinearLayer(
                    num_classes=out_dim,  use_bias=False, use_bn=True, name='l_0')
            ]
        elif FLAGS.proj_head_mode == 'nonlinear':
            if FLAGS.num_proj_layers > 2:
                for j in range(FLAGS.num_proj_layers):
                    if j == 0:
                        self.linear_layers.append(
                            modify_LinearLayer(
                                num_classes=lambda input_shape: int(
                                    input_shape[-1]),
                                up_scale=FLAGS.up_scale, non_contrastive=FLAGS.non_contrastive,
                                use_bias=True,
                                use_bn=True,
                                name='nl_%d' % j))

                    elif j != FLAGS.num_proj_layers - 1:
                        # for the middle layers, use bias and relu for the output.
                        if FLAGS.reduce_linear_dimention:
                            print("Implement reduce dimention")
                            self.linear_layers.append(
                                modify_LinearLayer(
                                    num_classes=lambda input_shape: int(
                                        input_shape[-1]/2),
                                    up_scale=FLAGS.up_scale, non_contrastive=False,
                                    use_bias=True,
                                    use_bn=True,
                                    name='nl_%d' % j))
                        else:
                            self.linear_layers.append(
                                modify_LinearLayer(
                                    num_classes=lambda input_shape: int(
                                        input_shape[-1]),
                                    up_scale=FLAGS.up_scale, non_contrastive=False,
                                    use_bias=True,
                                    use_bn=True,
                                    name='nl_%d' % j))

                    else:
                        # for the final layer, neither bias nor relu is used.
                        self.linear_layers.append(
                            modify_LinearLayer(
                                num_classes=FLAGS.proj_out_dim,
                                use_bias=False,
                                use_bn=True,
                                name='nl_%d' % j))

            else:
                for j in range(FLAGS.num_proj_layers):
                    if j != FLAGS.num_proj_layers - 1:
                        # for the middle layers, use bias and relu for the output.
                        self.linear_layers.append(
                            modify_LinearLayer(
                                num_classes=lambda input_shape: int(
                                    input_shape[-1]),
                                up_scale=FLAGS.up_scale, non_contrastive=FLAGS.non_contrastive,
                                use_bias=True,
                                use_bn=True,
                                name='nl_%d' % j))
                    else:
                        # for the final layer, neither bias nor relu is used.
                        self.linear_layers.append(
                            modify_LinearLayer(
                                num_classes=FLAGS.proj_out_dim,
                                up_scale=FLAGS.up_scale, non_contrastive=False,
                                use_bias=False,
                                use_bn=True,
                                name='nl_%d' % j))
        else:
            raise ValueError('Unknown head projection mode {}'.format(
                FLAGS.proj_head_mode))

        super(PredictionHead, self).__init__(**kwargs)

    def call(self, inputs, training):
        if FLAGS.proj_head_mode == 'none':
            return inputs  # directly use the output hiddens as hiddens
        hiddens_list = [tf.identity(inputs, 'proj_head_input')]
        if FLAGS.proj_head_mode == 'linear':
            assert len(self.linear_layers) == 1, len(self.linear_layers)
            return hiddens_list.append(self.linear_layers[0](hiddens_list[-1],
                                                             training))

        elif FLAGS.proj_head_mode == 'nonlinear':
            for j in range(FLAGS.num_proj_layers):
                hiddens = self.linear_layers[j](hiddens_list[-1], training)
                if j != FLAGS.num_proj_layers - 1:
                    # for the middle layers, use bias and relu for the output.
                    hiddens = tf.nn.relu(hiddens)
                hiddens_list.append(hiddens)

        else:
            raise ValueError('Unknown head projection mode {}'.format(
                FLAGS.proj_head_mode))

        # The first element is the output of the projection head.
        # The second element is the input of the finetune head.
        proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')
        return proj_head_output


# ******************************************************************
# Non Contrastive Framework Models
# ******************************************************************
'''Noted this Design Using Resnet from SimCLR --> Not modify version
+ ResNet Modify Version will Control output Spatial Features Maps
Ex: (7*7, 14*14, 28*28,)*(1024 or 2048) Dimension 
--> 
'''
# implement with Standard ResNet Ouput


class online_model(tf.keras.models.Model):
    """Resnet model with projection or supervised layer."""

    def __init__(self, num_classes, **kwargs):

        super(online_model, self).__init__(**kwargs)
        # Encoder
        # self.resnet_model = resnet.resnet(
        #     resnet_depth=FLAGS.resnet_depth,
        #     width_multiplier=FLAGS.width_multiplier,
        #     cifar_stem=FLAGS.image_size <= 32)
        self.resnet_model = resnet_modify(resnet_depth=FLAGS.resnet_depth,
                                         width_multiplier=FLAGS.width_multiplier)
        # Projcetion head
        self._projection_head = ProjectionHead()
        self.globalaveragepooling = tf.keras.layers.GlobalAveragePooling2D()

        # Supervised classficiation head
        if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
            self.supervised_head = SupervisedHead(num_classes)

    def __call__(self, inputs, training):

        features = inputs

        if training and FLAGS.train_mode == 'pretrain':
            if FLAGS.fine_tune_after_block > -1:
                raise ValueError('Does not support layer freezing during pretraining,'
                                 'should set fine_tune_after_block<=-1 for safety.')

        if inputs.shape[3] is None:
            raise ValueError('The input channels dimension must be statically known '
                             f'(got input shape {inputs.shape})')

        # # Base network forward pass.
        hiddens = self.resnet_model(features, training=training)
        hiddens = self.globalaveragepooling(hiddens)
        #print("Output from ResNet Model", hiddens.shape)
        # Add heads.
        projection_head_outputs, supervised_head_inputs, = self._projection_head(
            hiddens, training)

        #print("output from  Online projection Head",projection_head_outputs.shape )

        if FLAGS.train_mode == 'finetune':
            supervised_head_outputs = self.supervised_head(supervised_head_inputs,
                                                           training)
            return None, supervised_head_outputs

        elif FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            # When performing pretraining and linear evaluation together we do not
            # want information from linear eval flowing back into pretraining network
            # so we put a stop_gradient.
            supervised_head_outputs = self.supervised_head(
                tf.stop_gradient(supervised_head_inputs), training)
            #print("Supervised Head Output Dim", supervised_head_outputs.shape)
            return projection_head_outputs, supervised_head_outputs

        else:
            return projection_head_outputs, None

# Consideration take Supervised evaluate From the Target model


class target_model(tf.keras.models.Model):
    """Resnet model with projection or supervised layer."""

    def __init__(self, num_classes, **kwargs):

        super(target_model, self).__init__(**kwargs)
        # Encoder
        # self.resnet_model = resnet.resnet(
        #     resnet_depth=FLAGS.resnet_depth,
        #     width_multiplier=FLAGS.width_multiplier,
        #     cifar_stem=FLAGS.image_size <= 32)
        self.resnet_model = resnet_modify(resnet_depth=FLAGS.resnet_depth,
                                         width_multiplier=FLAGS.width_multiplier)
        # Projcetion head
        self._projection_head = ProjectionHead()
        self.globalaveragepooling = tf.keras.layers.GlobalAveragePooling2D()

        # Supervised classficiation head
        if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
            self.supervised_head = SupervisedHead(num_classes)

    def __call__(self, inputs, training):

        features = inputs

        if training and FLAGS.train_mode == 'pretrain':
            if FLAGS.fine_tune_after_block > -1:
                raise ValueError('Does not support layer freezing during pretraining,'
                                 'should set fine_tune_after_block<=-1 for safety.')

        if inputs.shape[3] is None:
            raise ValueError('The input channels dimension must be statically known '
                             f'(got input shape {inputs.shape})')

        # # Base network forward pass.
        hiddens = self.resnet_model(features, training=training)
        hiddens = self.globalaveragepooling(hiddens)

        # Add heads.
        projection_head_outputs, supervised_head_inputs = self._projection_head(
            hiddens, training)

        if FLAGS.train_mode == 'finetune':
            supervised_head_outputs = self.supervised_head(supervised_head_inputs,
                                                           training)
            return None, supervised_head_outputs

        elif FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            # When performing pretraining and linear evaluation together we do not
            # want information from linear eval flowing back into pretraining network
            # so we put a stop_gradient.
            supervised_head_outputs = self.supervised_head(
                tf.stop_gradient(supervised_head_inputs), training)

            return projection_head_outputs, supervised_head_outputs

        else:
            return projection_head_outputs, None

# Implement with Encode ResNet Modify output Spatial Feature Map (SIZE, Channels)
'''
1. Control the stride output
2. Global Average spatial feature map --> Feeding NLP
Update requirement 
+ Setting FLAGS control spatial output (control by FLAGS.Stride)
+ Setting FLAGS control the middel layers Output if Necessary

'''


class online_model_v1(tf.keras.models.Model):
    """Resnet modify model with projection or supervised layer."""

    def __init__(self, num_classes, **kwargs):

        super(online_model_v1, self).__init__(**kwargs)

        # Encoder using Resnet with Larger Output spatial Dimension
        self.resnet_model = resnet_modify.resnet(
            resnet_depth=FLAGS.resnet_depth,
            width_multiplier=FLAGS.width_multiplier,
            cifar_stem=FLAGS.image_size <= 32)

        # Projcetion head
        self._projection_head = ProjectionHead()
        # This implementation when using modify Resnet
        self.globalaveragepooling = tf.keras.layers.GlobalAveragePooling2D()

        # Supervised classficiation head
        if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
            self.supervised_head = SupervisedHead(num_classes)

    def __call__(self, inputs, training):

        features = inputs

        if training and FLAGS.train_mode == 'pretrain':
            if FLAGS.fine_tune_after_block > -1:
                raise ValueError('Does not support layer freezing during pretraining,'
                                 'should set fine_tune_after_block<=-1 for safety.')

        if inputs.shape[3] is None:
            raise ValueError('The input channels dimension must be statically known '
                             f'(got input shape {inputs.shape})')

        # network forward pass.
        # different spatial feature outputs
        hiddens = self.resnet_model(features, training=training)
        hiddens = self.globalaveragepooling(hiddens)

        # Add heads.
        projection_head_outputs, supervised_head_inputs, = self._projection_head(
            hiddens, training)

        if FLAGS.train_mode == 'finetune':
            supervised_head_outputs = self.supervised_head(supervised_head_inputs,
                                                           training)
            return None, supervised_head_outputs

        elif FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            # When performing pretraining and linear evaluation together we do not
            # want information from linear eval flowing back into pretraining network
            # so we put a stop_gradient.
            supervised_head_outputs = self.supervised_head(
                tf.stop_gradient(supervised_head_inputs), training)
            #print("Supervised Head Output Dim", supervised_head_outputs.shape)
            return projection_head_outputs, supervised_head_outputs

        else:
            return projection_head_outputs, None

class target_model_v1(tf.keras.models.Model):

    def __init__(self, num_classes, **kwargs):

        super(target_model_v1, self).__init__(**kwargs)
        # Encoder
        self.resnet_model = resnet_modify.resnet(
            resnet_depth=FLAGS.resnet_depth,
            width_multiplier=FLAGS.width_multiplier,
            cifar_stem=FLAGS.image_size <= 32)

        # Projcetion head
        self._projection_head = ProjectionHead()
        # This implementation when using modify Resnet
        self.globalaveragepooling = tf.keras.layers.GlobalAveragePooling2D()

        # Supervised classficiation head
        if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
            self.supervised_head = SupervisedHead(num_classes)

    def __call__(self, inputs, training):

        features = inputs

        if training and FLAGS.train_mode == 'pretrain':
            if FLAGS.fine_tune_after_block > -1:
                raise ValueError('Does not support layer freezing during pretraining,'
                                 'should set fine_tune_after_block<=-1 for safety.')

        if inputs.shape[3] is None:
            raise ValueError('The input channels dimension must be statically known '
                             f'(got input shape {inputs.shape})')

        # # Base network forward pass.
        hiddens = self.resnet_model(features, training=training)
        hiddens = self.globalaveragepooling(hiddens)

        # Add heads.
        projection_head_outputs, supervised_head_inputs = self._projection_head(
            hiddens, training)

        if FLAGS.train_mode == 'finetune':
            supervised_head_outputs = self.supervised_head(supervised_head_inputs,
                                                           training)
            return None, supervised_head_outputs

        elif FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            # When performing pretraining and linear evaluation together we do not
            # want information from linear eval flowing back into pretraining network
            # so we put a stop_gradient.
            supervised_head_outputs = self.supervised_head(
                tf.stop_gradient(supervised_head_inputs), training)

            return projection_head_outputs, supervised_head_outputs

        else:
            return projection_head_outputs, None


# ******************************************************************
# Contrastive Framework Models
# ******************************************************************
'''
For Contrastive Framework --> 
We might Design MoCo Contrastive Framework instead of SimCLR
'''

class contrast_models(tf.keras.models.Model):
    """Resnet model with projection or supervised layer."""

    def __init__(self, num_classes, **kwargs):

        super(contrast_models, self).__init__(**kwargs)
        # Encoder
        self.resnet_model = resnet.resnet(
            resnet_depth=FLAGS.resnet_depth,
            width_multiplier=FLAGS.width_multiplier,
            cifar_stem=FLAGS.image_size <= 32)
        # Projcetion head
        self._projection_head = ProjectionHead()

        # Supervised classficiation head
        if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
            self.supervised_head = SupervisedHead(num_classes)

    def __call__(self, inputs, training):
        # print(inputs)
        features = inputs

        if training and FLAGS.train_mode == 'pretrain':
            if FLAGS.fine_tune_after_block > -1:
                raise ValueError('Does not support layer freezing during pretraining,'
                                 'should set fine_tune_after_block<=-1 for safety.')

        if inputs.shape[3] is None:
            raise ValueError('The input channels dimension must be statically known '
                             f'(got input shape {inputs.shape})')

        # # Base network forward pass.
        hiddens = self.resnet_model(features, training=training)

        # Add heads.
        projection_head_outputs, supervised_head_inputs = self._projection_head(
            hiddens, training)

        if FLAGS.train_mode == 'finetune':
            supervised_head_outputs = self.supervised_head(supervised_head_inputs,
                                                           training)
            return None, supervised_head_outputs

        elif FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            # When performing pretraining and linear evaluation together we do not
            # want information from linear eval flowing back into pretraining network
            # so we put a stop_gradient.
            supervised_head_outputs = self.supervised_head(
                tf.stop_gradient(supervised_head_inputs), training)

            return projection_head_outputs, supervised_head_outputs

        else:
            return projection_head_outputs, None
