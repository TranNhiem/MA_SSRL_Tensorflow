# Fast Auto Augument
import torch
import torchvision
from torchvision.transforms import transforms
from .searched_policies import fa_reduced_cifar10, fa_resnet50_rimagenet, fa_reduced_svhn
from .transform_table import augment_list
from PIL import Image
import numpy as np
import random


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies
        self.augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}
        self.trfs_info = {}

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = self.apply_augment(img, name, level)
        return img
    
    def apply_augment(self, img, name, level):
        augment_fn, low, high = self.augment_dict[name]
        self.trfs_info[name] = degree = level * (high - low) + low
        return augment_fn(img.copy(), degree) 


class Fast_AutoAugment(object):

    def __init__(self, policy_type="imagenet"):
        # preprocess..
        self.trfs_cntr = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
        ])

        if policy_type == "imagenet":
            ds_policies = Augmentation(fa_resnet50_rimagenet())
        elif policy_type == "redu_cifar10": 
            ds_policies = Augmentation(fa_reduced_cifar10())
        elif policy_type == "redu_svhn":
            ds_policies = Augmentation(fa_reduced_svhn())
        else:
            raise ValueError("The policies of indicated dataset have not been searched")
        
        self.policy_type = policy_type
        self.trfs_cntr.transforms.insert(-1, ds_policies)
        self.policy_info = ds_policies.trfs_info

    def prnt_policies(self):
        if self.policy_type == "imagenet":
            ds_policies = fa_resnet50_rimagenet()
        elif self.policy_type == "redu_cifar10": 
            ds_policies = fa_reduced_cifar10()
        elif self.policy_type == "redu_svhn":
            ds_policies = fa_reduced_svhn()

        return ds_policies

    def distort(self, image):
        # dummy transformation of tf.tensor & PIL
        pil_im = Image.fromarray( image.numpy().astype(np.uint8) )
        da_ims = self.trfs_cntr(pil_im)
        # PIL format have different channel order from tf.tensor : e.g. (-, c(3), h, w) -> (-, h, w, c(3)), batch channel omit..
        da_ims = np.einsum('chw->hwc', da_ims)
        return da_ims, [*self.policy_info.items()]


# sample code snippet..
if __name__ == '__main__':
    import numpy as np
    img = np.random.random((14, 14, 3))

    fa = Fast_AutoAugment()
    #print(fa.prnt_policies()[0])
    print(fa.distort(img))