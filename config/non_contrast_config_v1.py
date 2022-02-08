from .absl_mock import Mock_Flag
from .config_non_contrast import read_cfg


def read_cfg_base(mod="non_contrastive"):
    flag = read_cfg(mod)
    FLAGS = flag.FLAGS

    print(f"\n\n Great! seems you import the code under {__file__} \n\n")
    '''ATTENTION --> Changing the training_loop FLAGS Corresponding" '''
    FLAGS.training_loop = "two_views"  # ['two_views', "multi_views", ]

    # , ['ds_1_2_options', 'train_ds_options'],
    FLAGS.dataloader = True
    FLAGS.mode_prefetch = 25  # if set it to 1 will Use AUTO
    # ["custome", "TFA_API"] # Current suport TFA_API
    FLAGS.auto_augment = "custome"
    # set True will resize inside wrap_ds else resize in Wrap_da STEP
    FLAGS.resize_wrap_ds = True

    FLAGS.wandb_project_name = "mutli_augmentation_strategies"
    FLAGS.wandb_run_name = "Res-18_RandCrop_AutoAugment_V1_30per_trainset"
    FLAGS.wandb_mod = "dryrun" #"online"
    FLAGS.restore_checkpoint = False  # Restore Checkpoint or Not

    '''
        The middle layer output control the feature map size
    '''
    FLAGS.Middle_layer_output = None
    FLAGS.original_loss_stop_gradient = False
    FLAGS.Encoder_block_strides = {'1': 2, '2': 1, '3': 2, '4': 2, '5': 2}

    FLAGS.Encoder_block_channel_output = {
        '1': 1, '2': 1, '3': 1, '4': 1, '5': 1}

    # byol_asymmetrized_loss (2 options --> Future Update with Mixed Loss)
    FLAGS.loss_type = "byol_symmetrized_loss"
    FLAGS.Loss_global_local ="cos_schedule" # cos_schedule or {passing any strings}
    FLAGS.alpha_base=0.7 # The base_value of Alpha
    FLAGS.alpha= 0.8 # Alpha values is  weighted loss between (Global and Local) Views
    # two options [fixed_value, schedule] schedule recommend from BYOL
    FLAGS.moving_average = "schedule"
    # ['fp16', 'fp32'],  # fp32 is original precision

    FLAGS.mixprecision = 'FP32'
    # , [ 'original', 'model_only', ],
    FLAGS.XLA_compiler = "original"
    FLAGS.base_lr = 0.3

    FLAGS.resnet_depth = 18
    FLAGS.train_epochs = 100
    FLAGS.num_classes = 1000

    FLAGS.train_batch_size = 128
    FLAGS.val_batch_size = 128
    #FLAGS.model_dir = "./model_ckpt/autoDA"
    #FLAGS.model_dir = "./model_ckpt/testMV"
    FLAGS.model_dir = "/data/multi_augment_project/resnet_byol/ResNet18_RandCrop_AutoAugment_V1_30per_trainset"
    #FLAGS.train_mode = "finetune"

    # ds ratio 
    FLAGS.tra_ds_ratio = 40
    FLAGS.subset_percentage = 0.3
    FLAGS.n_cls = FLAGS.num_classes

    return flag
