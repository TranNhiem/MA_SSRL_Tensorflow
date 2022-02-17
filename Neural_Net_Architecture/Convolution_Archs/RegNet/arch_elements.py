import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
import imagenet_utils as img_utils
from tensorflow.keras import utils as tf_utils

from metadata import WEIGHTS_HASHES, BASE_WEIGHTS_PATH

## I. Basic element of RegNet : 
def PreStem(name=None):
    """Rescales and normalizes inputs to [0,1] and ImageNet mean and std.
    Args:
    name: name prefix
    Returns:
    Rescaled and normalized tensor
    """
    if name is None:
        name = "prestem" + str(backend.get_uid("prestem"))

    def apply(x):
        x = layers.Rescaling(scale=1. / 255., name=name + "_prestem_rescaling")(x)
        return x

    return apply


def Stem(name=None):
    """Implementation of RegNet stem.
    (Common to all model variants)
    Args:
    name: name prefix
    Returns:
    Output tensor of the Stem
    """
    if name is None:
        name = "stem" + str(backend.get_uid("stem"))

    def apply(x):
        x = layers.Conv2D(
            32, (3, 3),
            strides=2,
            use_bias=False,
            padding="same",
            kernel_initializer="he_normal",
            name=name + "_stem_conv")(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_stem_bn")(x)
        x = layers.ReLU(name=name + "_stem_relu")(x)
        return x

    return apply


def SqueezeAndExciteBlock(filters_in, se_filters, name=None):
    """Implements the Squeeze and excite block (https://arxiv.org/abs/1709.01507).
    Args:
    filters_in: input filters to the block
    se_filters: filters to squeeze to
    name: name prefix
    Returns:
    A function object
    """
    if name is None:
        name = str(backend.get_uid("squeeze_and_excite"))

    def apply(inputs):
        x = layers.GlobalAveragePooling2D(
            name=name + "_squeeze_and_excite_gap", keepdims=True)(inputs)
        x = layers.Conv2D(
            se_filters, (1, 1),
            activation="relu",
            kernel_initializer="he_normal",
            name=name + "_squeeze_and_excite_squeeze")(x)
        x = layers.Conv2D(
            filters_in, (1, 1),
            activation="sigmoid",
            kernel_initializer="he_normal",
            name=name + "_squeeze_and_excite_excite")(x)
        x = tf.math.multiply(x, inputs)
        return x

    return apply


def XBlock(filters_in, filters_out, group_width, stride=1, name=None):
    """Implementation of X Block.
    Reference: [Designing Network Design
    Spaces](https://arxiv.org/abs/2003.13678)
    Args:
    filters_in: filters in the input tensor
    filters_out: filters in the output tensor
    group_width: group width
    stride: stride
    name: name prefix
    Returns:
    Output tensor of the block
    """
    if name is None:
        name = str(backend.get_uid("xblock"))

    def apply(inputs):
        if filters_in != filters_out and stride == 1:
            raise ValueError(
                f"Input filters({filters_in}) and output filters({filters_out}) "
                f"are not equal for stride {stride}. Input and output filters must "
                f"be equal for stride={stride}.")

        # Declare layers
        groups = filters_out // group_width

        if stride != 1:
            skip = layers.Conv2D(
                filters_out, (1, 1),
                strides=stride,
                use_bias=False,
                kernel_initializer="he_normal",
                name=name + "_skip_1x1")(inputs)
            skip = layers.BatchNormalization(
                momentum=0.9, epsilon=1e-5, name=name + "_skip_bn")(skip)
        else:
            skip = inputs

        # Build block
        # conv_1x1_1
        x = layers.Conv2D(
            filters_out, (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_1")(inputs)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_1_bn")(x)
        x = layers.ReLU(name=name + "_conv_1x1_1_relu")(x)

        # conv_3x3
        x = layers.Conv2D(
            filters_out, (3, 3),
            use_bias=False,
            strides=stride,
            groups=groups,
            padding="same",
            kernel_initializer="he_normal",
            name=name + "_conv_3x3")(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_3x3_bn")(x)
        x = layers.ReLU(name=name + "_conv_3x3_relu")(x)

        # conv_1x1_2
        x = layers.Conv2D(
            filters_out, (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_2")(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_2_bn")(x)

        x = layers.ReLU(name=name + "_exit_relu")(x + skip)

        return x

    return apply


def YBlock(filters_in,
           filters_out,
           group_width,
           stride=1,
           squeeze_excite_ratio=0.25,
           name=None):
    """Implementation of Y Block.
    Reference: [Designing Network Design
    Spaces](https://arxiv.org/abs/2003.13678)
    Args:
    filters_in: filters in the input tensor
    filters_out: filters in the output tensor
    group_width: group width
    stride: stride
    squeeze_excite_ratio: expansion ration for Squeeze and Excite block
    name: name prefix
    Returns:
    Output tensor of the block
    """
    if name is None:
        name = str(backend.get_uid("yblock"))

    def apply(inputs):
        if filters_in != filters_out and stride == 1:
            raise ValueError(
                f"Input filters({filters_in}) and output filters({filters_out}) "
                f"are not equal for stride {stride}. Input and output filters must  "
                f"be equal for stride={stride}.")

        groups = filters_out // group_width
        se_filters = int(filters_in * squeeze_excite_ratio)

        if stride != 1:
            skip = layers.Conv2D(
                filters_out, (1, 1),
                strides=stride,
                use_bias=False,
                kernel_initializer="he_normal",
                name=name + "_skip_1x1")(inputs)
            skip = layers.BatchNormalization(
                momentum=0.9, epsilon=1e-5, name=name + "_skip_bn")(skip)
        else:
            skip = inputs

        # Build block
        # conv_1x1_1
        x = layers.Conv2D(
            filters_out, (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_1")(inputs)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_1_bn")(x)
        x = layers.ReLU(name=name + "_conv_1x1_1_relu")(x)

        # conv_3x3
        x = layers.Conv2D(
            filters_out, (3, 3),
            use_bias=False,
            strides=stride,
            groups=groups,
            padding="same",
            kernel_initializer="he_normal",
            name=name + "_conv_3x3")(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_3x3_bn")(x)
        x = layers.ReLU(name=name + "_conv_3x3_relu")(x)

        # Squeeze-Excitation block
        x = SqueezeAndExciteBlock(filters_out, se_filters, name=name)(x)

        # conv_1x1_2
        x = layers.Conv2D(
            filters_out, (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_2")(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_2_bn")(x)

        x = layers.ReLU(name=name + "_exit_relu")(x + skip)

        return x

    return apply


# new feature add in keras-nightly implementation
def ZBlock(filters_in,
           filters_out,
           group_width,
           stride=1,
           squeeze_excite_ratio=0.25,
           bottleneck_ratio=0.25,
           name=None):
    """Implementation of Z block Reference: [Fast and Accurate Model Scaling](https://arxiv.org/abs/2103.06877).
    Args:
    filters_in: filters in the input tensor
    filters_out: filters in the output tensor
    group_width: group width
    stride: stride
    squeeze_excite_ratio: expansion ration for Squeeze and Excite block
    bottleneck_ratio: inverted bottleneck ratio
    name: name prefix
    Returns:
    Output tensor of the block
    """
    if name is None:
        name = str(backend.get_uid("zblock"))

    def apply(inputs):
        if filters_in != filters_out and stride == 1:
            raise ValueError(
                f"Input filters({filters_in}) and output filters({filters_out})"
                f"are not equal for stride {stride}. Input and output filters must be"
                f" equal for stride={stride}.")

        groups = filters_out // group_width
        se_filters = int(filters_in * squeeze_excite_ratio)

        inv_btlneck_filters = int(filters_out / bottleneck_ratio)

        # Build block
        # conv_1x1_1
        x = layers.Conv2D(
            inv_btlneck_filters, (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_1")(inputs)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_1_bn")(x)
        x = tf.nn.silu(x)

        # conv_3x3
        x = layers.Conv2D(
            inv_btlneck_filters, (3, 3),
            use_bias=False,
            strides=stride,
            groups=groups,
            padding="same",
            kernel_initializer="he_normal",
            name=name + "_conv_3x3")(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_3x3_bn")(x)
        x = tf.nn.silu(x)

        # Squeeze-Excitation block
        x = SqueezeAndExciteBlock(inv_btlneck_filters, se_filters, name=name)

        # conv_1x1_2
        x = layers.Conv2D(
            filters_out, (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_2")(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_2_bn")(x)

        if stride != 1:
            return x
        else:
            return x + inputs

    return apply


def Stage(block_type, depth, group_width, filters_in, filters_out, name=None):
    """Implementation of Stage in RegNet.
    Args:
    block_type: must be one of "X", "Y", "Z"
    depth: depth of stage, number of blocks to use
    group_width: group width of all blocks in  this stage
    filters_in: input filters to this stage
    filters_out: output filters from this stage
    name: name prefix
    Returns:
    Output tensor of Stage
    """
    if name is None:
        name = str(backend.get_uid("stage"))

    def apply(inputs):
        x = inputs
        if block_type == "X":
            x = XBlock(
                filters_in,
                filters_out,
                group_width,
                stride=2,
                name=f"{name}_XBlock_0")(x)
            for i in range(1, depth):
                x = XBlock(
                    filters_out, filters_out, group_width, name=f"{name}_XBlock_{i}")(x)
        elif block_type == "Y":
            x = YBlock(
                filters_in,
                filters_out,
                group_width,
                stride=2,
                name=name + "_YBlock_0")(x)
            for i in range(1, depth):
                x = YBlock(
                    filters_out, filters_out, group_width, name=f"{name}_YBlock_{i}")(x)
        elif block_type == "Z":
            x = ZBlock(
                filters_in,
                filters_out,
                group_width,
                stride=2,
                name=f"{name}_ZBlock_0")(x)
            for i in range(1, depth):
                x = ZBlock(
                    filters_out, filters_out, group_width, name=f"{name}_ZBlock_{i}")(x)
        else:
            raise NotImplementedError(f"Block type `{block_type}` not recognized."
                                    f"block_type must be one of (`X`, `Y`, `Z`). ")
        return x

    return apply


def Head(num_classes=1000, name=None):
    """Implementation of classification head of RegNet.
    Args:
    num_classes: number of classes for Dense layer
    name: name prefix
    Returns:
    Output logits tensor.
    """
    if name is None:
        name = str(backend.get_uid("head"))

    def apply(x):
        x = layers.GlobalAveragePooling2D(name=name + "_head_gap")(x)
        x = layers.Dense(num_classes, name=name + "head_dense")(x)
        return x

    return apply


## II. RegNet arch definition : 
def RegNet(depths,
           widths,
           group_width,
           block_type,
           default_size,
           model_name="regnet",
           include_preprocessing=True,
           include_top=False,     # In SSL-framework, it's disable by default
           weights=None,          # In SSL-framework, it's disable by default
           input_tensor=None,
           input_shape=None,
           pooling=None,
           classes=1000,
           classifier_activation="softmax"):
    """Instantiates RegNet architecture given specific configuration.
    Args:
    depths: An iterable containing depths for each individual stages.
    widths: An iterable containing output channel width of each individual
        stages
    group_width: Number of channels to be used in each group. See grouped
        convolutions for more information.
    block_type: Must be one of `{"X", "Y", "Z"}`. For more details see the
        papers "Designing network design spaces" and "Fast and Accurate Model
        Scaling"
    default_size: Default input image size.
    model_name: An optional name for the model.
    include_preprocessing: boolean denoting whther to include preprocessing in
        the model
    include_top: Boolean denoting whether to include classification head to the
        model.
    weights: one of `None` (random initialization), "imagenet" (pre-training on
        ImageNet), or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use
        as image input for the model.
    input_shape: optional shape tuple, only to be specified if `include_top` is
        False. It should have exactly 3 inputs channels.
    pooling: optional pooling mode for feature extraction when `include_top` is
        `False`. - `None` means that the output of the model will be the 4D tensor
        output of the last convolutional layer. - `avg` means that global average
        pooling will be applied to the output of the last convolutional layer, and
        thus the output of the model will be a 2D tensor. - `max` means that
        global max pooling will be applied.
    classes: optional number of classes to classify images into, only to be
        specified if `include_top` is True, and if no `weights` argument is
        specified.
    classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
    Returns:
    A `keras.Model` instance.
    Raises:
        ValueError: in case of invalid argument for `weights`,
        or invalid input shape.
        ValueError: if `classifier_activation` is not `softmax` or `None` when
        using a pretrained top layer.
        ValueError: if `include_top` is True but `num_classes` is not 1000.
        ValueError: if `block_type` is not one of `{"X", "Y", "Z"}`
    """
    if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
        raise ValueError("The `weights` argument should be either "
                            "`None` (random initialization), `imagenet` "
                            "(pre-training on ImageNet), "
                            "or the path to the weights file to be loaded.")

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError("If using `weights` as `'imagenet'` with `include_top`"
                            " as true, `classes` should be 1000")

    # Determine proper input shape
    input_shape = img_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights
    )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if input_tensor is not None:
        inputs = tf_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    x = inputs
    if include_preprocessing:
        x = PreStem(name=model_name)(x)
        x = Stem(name=model_name)(x)

    in_channels = 32  # Output from Stem

    for num_stage in range(4):
        depth = depths[num_stage]
        out_channels = widths[num_stage]

        x = Stage(
            block_type,
            depth,
            group_width,
            in_channels,
            out_channels,
            name=model_name + "_Stage_" + str(num_stage))(x)
        in_channels = out_channels

    if include_top:
        x = Head(num_classes=classes)(x)
        img_utils.validate_activation(classifier_activation, weights)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name=model_name)

    # Load weights.
    if weights == "imagenet":
        if include_top:
            file_suffix = ".h5"
            file_hash = WEIGHTS_HASHES[model_name[-4:]][0]
        else:
            file_suffix = "_notop.h5"
            file_hash = WEIGHTS_HASHES[model_name[-4:]][1]
        file_name = model_name + file_suffix
        weights_path = tf_utils.get_file(
            file_name,
            BASE_WEIGHTS_PATH + file_name,
            cache_subdir="models",
            file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model