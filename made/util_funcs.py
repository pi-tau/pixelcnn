from PIL import Image
import random

import numpy as np
import torch


def fix_random_seeds(seed):
    """Manually set the seed for random number generation.
    Also set CuDNN flags for reproducible results using deterministic algorithms.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_printoptions(precision, sci_mode):
    torch.set_printoptions(precision=precision, sci_mode=sci_mode)
    np.set_printoptions(precision=precision, suppress=~sci_mode)


def generate_2D_distr_from_img(image_file, d=0):
    """Given an image re-scale it, convert it to gray-scale, and threat the
    resulting pixel values as the shape of a 2D probability distribution.
    The resulting distribution gives the probability for each of the pixels
    being "on" or "off".

    Args:
        image_file (str): File path to the image.
        d (int): Dimension size. The image is re-scaled to size (d,d).
            Default value is 0, i.e. no resizing.

    Returns:
        distr (np.Array): Numpy array of shape (d, d) giving the "true"
            distribution generated from the pixel values of the image.
    """
    im = Image.open(image_file)
    if d > 0:
        im = im.resize(size=(d,d)) # re-scale to the requested size
    im = im.convert(mode="L")  # convert to gray-scale
    im = np.array(im).astype(np.float32)
    im = 255. - im

    distr = (im / im.sum()) # convert the pixel values to a distribution

    return distr

#