from config.absl_mock import Mock_Flag
from config.config_non_contrast import read_cfg


def read_cfg(mod="non_contrastive"):
    read_cfg(mod)
    flag = Mock_Flag()
    FLAGS = flag.FLAGS

    FLAGS.wandb_project_name = "mutli_augmentation_strategies"
    FLAGS.wandb_run_name = "multi_augmentation_autoaugment_rand_Crop"
    FLAGS.wandb_mod = "run"

    FLAGS.Middle_layer_output = None
    FLAGS.original_loss_stop_gradient = False

    FLAGS.Encoder_block_strides = {'1': 2, '2': 1, '3': 2, '4': 2, '5': 2}

    FLAGS.Encoder_block_channel_output = {
        '1': 1, '2': 1, '3': 1, '4': 1, '5': 1}

    # byol_asymmetrized_loss (2 options --> Future Update with Mixed Loss)
    FLAGS.loss_type = "byol_symmetrized_loss"
    # ['fp16', 'fp32'],  # fp32 is original precision
    FLAGS.mixprecision = 'fp32'
    FLAGS.base_lr = 0.3

    FLAGS.resnet_depth = 18
    FLAGS.train_epochs = 100
    FLAGS.num_classes = 100

    FLAGS.train_batch_size = 128
    FLAGS.val_batch_size = 128
    FLAGS.model_dir = "/data1/share/resnet_byol/resnet18/auto_augment_rand_crop"
    #FLAGS.train_mode = "finetune"
