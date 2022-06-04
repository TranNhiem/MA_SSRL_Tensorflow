## MULTI-AUGMENTATION FOR EFFICIENT VISUAL REPRESENTATION LEARNING FOR SELF-SUPERVISED PRE-TRAINING

<p align="center">
  <img src="images/MASSRL.gif" width="600">
</p>

MASSRL Official TensorFlow implementation of the [MASSRL paper](https://arxiv.org/abs/1911.05722):

<a target="[_parent](https://colab.research.google.com/drive/1fquGOr_psJfDXxOmdFVkfrbedGfi1t-X?usp=sharing)"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Visualization Multi-Augmentation Strategies on Google-Colab Notebook: https://colab.research.google.com/drive/1fquGOr_psJfDXxOmdFVkfrbedGfi1t-X?usp=sharing 

### Preparation for running on local Machines

We aims to be minimal modifications on code:

```
Install dependent packages following the [dev_requirement.txt] supporting tf_version_{2.4-> 2.9}. 

Downloading ImageNet-1K dataset (https://www.image-net.org/download.php).

pip install -e . for initialization the relative import path of .py modules

```

### Self-Supervised Pre-training

This implementation supports **Single-gpu**, **Multi-GPUs** training, DataParallel training is not supported.

To do self-superivsed pre-training of a ResNet-50 model on ImageNet in an (1-8)-gpus  following Three Stesp:

```
**1.Training Hyperparaneters Configures**: 

- you can change training hyperparameters setting (Dataset paths, All other training hyperperameters) base on
config/non_contrast_config_v1.py as Reference configure
- Consider you GPUs memmory >= 12G ResNet50 --> Recommend training on 4-> 8 GPUs.

**2.Execute MASSRL With 3 Augmentations Strategies SimCRL'Augmentation Pipeline, RandAug, AutoAugment**: 

-Nevigate to this directory
self_supervised_learning_frameworks/none_contrastive_framework/run_MASSRL.py
- Execute the 🏃‍♀️ file.
python run_MASSRL.py 
```

***Note***: for 8-gpus training, we recommend following the [linear lr scaling recipe](https://arxiv.org/abs/1706.02677): `--lr 0.2 --batch-size 128`. Other Hyperparameters can set default.
for 1-gpu training, we recommend following the [linear lr scaling recipe](https://arxiv.org/abs/1706.02677): `--lr 0.3 --batch-size 256`. Other Hyperparameters can set default.


## Downstream Tasks 

### Linear Image Classification
ImageNet Linear Evaluation

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 1-gpu machine, run:

```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

Linear classification results on ImageNet using this repo with 8 NVIDIA A100 GPUs :




### Models



### Transferring to Object Detection

See [./detection](detection).

## Citation for Our Paper
```
@Article{TranMASSRL,
  author  = {Van-Nhiem Tran, Chi-En Huang, Shen-Hsuan Liu, Kai-Lin Yang, Timothy Ko, Yung-Hui Li},
  title   = {Multi-Augmentation Strategies Disentangle represenation learning Self-Supervised},
  journal = {https://arxiv.org/abs/2205.11772},
  year    = {2022},
}
```

### Checkout Guideline for Contribution

Awesome! Thank You for being a part this project > > 
Before you start to contribute for this repository, please quick go through [Guidelines](contribution_guideline.md).

### License
This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### See Also

* [MASSRL.Pytorch-lightning](https://github.com/TranNhiem/MA_SSRL_Pytorch): A Pytorch-Lightning official implementation.
* [Colab notebook]([https:/test.ipynb](https://colab.research.google.com/drive/1fquGOr_psJfDXxOmdFVkfrbedGfi1t-X?usp=sharing): Multi-Augmentation demo on Colab GPU.

