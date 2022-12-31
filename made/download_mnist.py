import os
import pickle
import shutil
import struct

import numpy as np
import torchvision


def load_data(ROOT):
    """Load all of MNIST."""

    # The images should be downloaded from `yann.lecun.com/exdb/mnist`.
    # There are 4 files:
    #   * train-images-idx3-ubyte: training set images
    #   * train-images-idx1-ubyte: training set labels
    #   * t10k-images-idx3-ubyte: test set images
    #   * t10k-labels-idx1-ubyte: test set labels
    #
    # The idx file format is a simple format for vectors and multidimensional
    # matrices. The format is:
    #   magic number
    #   size in dimension 0
    #   size in dimension 1
    #   ...
    #   size in dimension N
    #   data
    # The magic number and the sizes are 4-byte integers. The magic number
    # stores information about the data type and the number of dimensions.
    # However we already know the number of dimensions and the data type so we
    # do not need to decode the magic number.

    # The image files have three dimensions -- number of images, width, and
    # height. Thus, four numbers have to be read from the file before the actual
    # data.
    with open(os.path.join(ROOT, "train-images-idx3-ubyte"), "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        X_train = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)

    with open(os.path.join(ROOT, "t10k-images-idx3-ubyte"), "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        X_test = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)

    # The labels have only one dimension -- number of labels. Thus, we need to
    # read two numbers before reading the actual data.
    with open(os.path.join(ROOT, "train-labels-idx1-ubyte"), "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        y_train = np.fromfile(f, dtype=np.uint8)

    with open(os.path.join(ROOT, "t10k-labels-idx1-ubyte"), "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        y_test = np.fromfile(f, dtype=np.uint8)

    # A list of classification classes.
    classes = np.array(np.arange(10))

    return X_train, y_train, X_test, y_test, classes


if __name__ == "__main__":
    root = "dataset"

    _ = torchvision.datasets.MNIST(root, download=True,)

    mnist_folder =  os.path.join(root, "MNIST")
    tmp = os.path.join(mnist_folder, "raw") # the folder where the MNIST images will be downloaded
    X_train, y_train, X_test, y_test, classes = load_data(tmp)

    with open(os.path.join(root, "mnist.pkl"), "wb") as f:
        pickle.dump({
            "X_train": X_train,
            "y_train": y_train,
            "X_test" : X_test,
            "y_test" : y_test,
            "classes": classes,
        }, f)

    # Remove the downloaded files.
    shutil.rmtree(mnist_folder)

#