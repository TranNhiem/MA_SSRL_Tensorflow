## MULTI-AUGMENTATION FOR EFFICIENT VISUAL REPRESENTATION LEARNING FOR SELF-SUPERVISED PRE-TRAINING

<p align="center">
  <img src="images/MASSRL.gif" width="600">
</p>

This repo is official TensorFlow implementation MASSRL.

[MASSRL Paper link](https://arxiv.org/abs/1911.05722)

[Blog Post]("Coming Soon")

This repo contains the source code for the `MASSRL` multi-Augmentation Strategies in Tensorflow models effortless and less error-prone.

## Table of Contents

  - [Installation](#installation)
  - [Visualization `MASSRL` Multi-Augmentation Strategies ](#Different-Multi-Augmentation-properties)
  - [Configure Self-Supervised Pretraining](#Setup-self-supervised-pretraining)
    - [Dataset](#Natural-Image-Dataset)
    - [Hyperamters Setting](#Important-Hyperparameter-Setting)
    - [Choosing # augmentation Strategies](#Number-Augmentation-Strategies)
    - [Single or Multi GPUs](#Single-Multi-GPUS)
  - [Contributing](#contributing)

## Installation

```
pip or conda installs these dependents in your local machine
```
* tensorflow==2.7.0, tensorflow-addons==0.15.0, tensorflow-datasets==4.4.0, tensorflow-estimator==2.7.0
* tqdm
* wandb
* imgaug

## Visualization `MASSRL` Multi-Augmentation Strategies

<a target="[_parent](https://colab.research.google.com/drive/1fquGOr_psJfDXxOmdFVkfrbedGfi1t-X?usp=sharing)"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Visualization Multi-Augmentation Strategies on Google-Colab Notebook: https://colab.research.google.com/drive/1fquGOr_psJfDXxOmdFVkfrbedGfi1t-X?usp=sharing 

Note the Visualization Augmentation *do not need to be trained* --- we are only Visualize Image after apply different Augmentation transformations.
However, you need to make sure that the `dataset` is appropriately passed down to the constructor of all submodules.
If you want to see this happen, please upvote [this Repo issue]


## Configuration Self-Supervised Pretraining

This implementation supports **Single-gpu**, **Multi-GPUs** training.

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

### Dataset 
**Note:** Public ImageNet dataset is implement in this work, if you have your own dataset you can change the path corresponding. 

#### Downloading ImageNet-1K dataset (https://www.image-net.org/download.php).

#### Using your own dataset 

Consider dataset folder structure setup for Pytorch `ImageFolder` and `DataLoader`principle

#### Changing dataset path(your path) in pretraining Flags: 

### Hyperparameter Setting 
You can change 

### Number Augmentation Strategies Implementation
You can select the number of Augmentation you want by setting the Flags

### Training Single or Multiple GPUs
You can set how many GPUs for training by changing the Flags
We will Support distributed multi-nodes(machines) traning soon 


### Models

### Transferring to Object Detection

See [./detection](detection).


## Checkout Guideline for Contribution

Awesome! Thank You for being a part this project > > 
Before you start to contribute for this repository, please quick go through [Guidelines](contribution_guideline.md).


### See Also

* [MASSRL.Pytorch-lightning](https://github.com/TranNhiem/MA_SSRL_Pytorch): A Pytorch-Lightning official implementation.

## Citation for Our Paper
```
@Article{TranMASSRL,
  author  = {Van-Nhiem Tran, Chi-En Huang, Shen-Hsuan Liu, Kai-Lin Yang, Timothy Ko, Yung-Hui Li},
  title   = {Multi-Augmentation Strategies Disentangle represenation learning Self-Supervised},
  journal = {https://arxiv.org/abs/2205.11772},
  year    = {2022},
}
```

