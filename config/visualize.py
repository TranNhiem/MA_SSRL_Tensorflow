import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os

import math
import errno
import shutil
from .helper_functions import *

class Visualize:
    def __init__(self,epoch,visualize_dir):
        self.epoch = epoch
        self.visualize_dir = visualize_dir

    def plot_feature_map(self,epoch,features):
        b,h,w,d= features.shape
        square = 10
        ix = 1
        for i in range(square):
            for j in range(square):
                ax = plt.subplot(square, square, ix)
                f = features[0,:,:,ix-1]
                f_max = np.max(f)
                f_min = np.min(f)
                f = (((f-f_min)/(f_max-f_min))*255.0).astype(np.uint8)
                # f = np.resize(f,(100,100))
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(f, cmap='gray')
                ix += 1
        plt.savefig(os.path.join(self.visualize_dir,str(epoch)+".png"))
        print("save in : ",os.path.join(self.visualize_dir,str(epoch)+".png"))





