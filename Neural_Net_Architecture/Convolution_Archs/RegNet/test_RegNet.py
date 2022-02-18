# tensorflow testing suits
from absl.testing import parameterized
from inspect import getmembers, isfunction
import numpy as np

import tensorflow as tf
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations

# RegNet pkg
import model_interface
from metadata import MODEL_CONFIGS, TEST_REGNET_PARAMS

## For now, the testing should be well-defined :
# example source : https://github.com/tensorflow/models/blob/master/official/vision/beta/modeling/backbones/resnet_test.py#L35

class RegNetTest(parameterized.TestCase, tf.test.TestCase):
    ## Testing Specification :
    def test_model_creation(self, input_size=224, endpoint_filter_scale=1):
        """Test creation of RegNet family models."""
        tf.keras.backend.set_image_data_format('channels_last')
        input_shape = (input_size, input_size, 3)

        # skip RegNet 'base' function, and grap all interface of RegNet
        interface_lst = getmembers(model_interface, isfunction)[1:]
        for mod_name, interface in interface_lst:
            mod = interface(input_shape=input_shape, include_top=False, weights=None, include_preprocessing=False)
            
            # 1. chk the number of model parameters, HACKME : modified the TEST_REGNET_PARAMS
            fail_msg = "model parameters should not beyond the given upper-bound"
            self.assertGreater(TEST_REGNET_PARAMS[mod_name], mod.count_params(), fail_msg)

            # 2. chk the shape of output feature map
            inputs = tf.keras.Input(shape=input_shape, batch_size=1)
            endpoints = mod(inputs)
            self.assertAllEqual(
                [1, input_size / 2**2, input_size / 2**2, 64 * endpoint_filter_scale],
                endpoints.shape.as_list()
            )
    
    '''
    @parameterized.parameters(
        (224, 1),
    )
    def test_sync_bn_multiple_devices(self, strategy, use_sync_bn):
        ...

    def test_resnet_rs(self):
        ...

    def test_input_specs(self, input_dim):
        ...

    def test_serialize_deserialize(self): # should move dict out of func
        # Create a network object that sets all of its config options.
        kwargs = dict(
            model_id=50,
            depth_multiplier=1.0,
            stem_type='v0',
            se_ratio=None,
            resnetd_shortcut=False,
            replace_stem_max_pool=False,
            init_stochastic_depth_rate=0.0,
            scale_stem=True,
            use_sync_bn=False,
            activation='relu',
            norm_momentum=0.99,
            norm_epsilon=0.001,
            kernel_initializer='VarianceScaling',
            kernel_regularizer=None,
            bias_regularizer=None,
            bn_trainable=True
        )
        network = resnet.ResNet(**kwargs)

        expected_config = dict(kwargs)
        self.assertEqual(network.get_config(), expected_config)

        # Create another network object from the first object's config.
        new_network = resnet.ResNet.from_config(network.get_config())

        # Validate that the config can be forced to JSON.
        _ = new_network.to_json()

        # If the serialization was successful, the new config should match the old.
        self.assertAllEqual(network.get_config(), new_network.get_config())
    '''

if __name__ == "__main__":
    tf.test.main()
