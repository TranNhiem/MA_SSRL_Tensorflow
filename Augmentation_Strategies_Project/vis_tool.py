from matplotlib import pyplot as plt
from math import sqrt, ceil
import numpy as np

def imshow_imgs(img_arr:np.ndarray, *args, **kwargs):
    def chk_args():
        assert isinstance(img_arr, np.ndarray)
        #assert img_arr.shape[-1] == len(titles)
        if len(img_arr.shape) > 4:
            raise ValueError("The image tensor should not greater then 4 channel (H, W, Channel, Batch)")
    
    chk_args()
    batch_size = img_arr.shape[0]
    if batch_size == 1:
        plt.imshow(img_arr[0], *args, **kwargs)

    row_siz = col_siz = ceil(sqrt(batch_size))
    fig, axes = plt.subplots(row_siz, col_siz)
    titles = [ idx for idx in range(batch_size)]
    # row first mode layout
    for idx, img in enumerate(img_arr, 0):
        (row_idx, col_idx) = divmod(idx, row_siz)
        axes[row_idx, col_idx].imshow(img_arr[idx], *args, **kwargs)
        axes[row_idx, col_idx].set_title(titles[idx])
        axes[row_idx, col_idx].axis('on')
    
    plt.show()
    