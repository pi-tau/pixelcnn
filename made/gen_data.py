import pickle

import numpy as np
from PIL import Image


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


if __name__ == "__main__":
    p = generate_2D_distr_from_img("img/smiley.jpg")#, d=25)
    with open("data/smiley.pkl", "wb") as f:
        pickle.dump(p, f)

    p = generate_2D_distr_from_img("img/horse.jpg", d=400)
    with open("data/horse.pkl", "wb") as f:
        pickle.dump(p, f)

    p = generate_2D_distr_from_img("img/new_york.jpg")#, d=400)
    with open("data/new_york.pkl", "wb") as f:
        pickle.dump(p, f)

#