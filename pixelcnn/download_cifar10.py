import os
import pickle
import shutil

import numpy as np
import torchvision


def load_data(ROOT):
    """Load all of CIFAR-10."""
    xs, ys = [], []

    # The archive contains the files data_batch_1, ..., data_batch_5, as well as
    # test_batch. Each of these files is a Python "pickled" object
    for _batch in range(1,6):
        with open(os.path.join(ROOT, "data_batch_%d" % (_batch)), "rb") as file:
            datadict = pickle.load(file, encoding="latin1")
            # Loaded in this way, each of the batch files contains a dictionary
            # with the following elements:
            # - data - a 10000x3072 numpy array. Each row of the array stores a
            #   32x32 color image. The first 1024 entries contain the red channel
            #   values, the next 1024 the green, and the final 1024 the blue.
            #   The image is stored in a row-major order, so that the first 32
            #   entries of the array are the red channel values of the first row
            #   of the image.
            #
            # - labels - a list of 10000 numbers in the range 0-9. The number at
            #   index i indicates the label of the i-th image in the array data.
            X = datadict["data"]
            Y = datadict["labels"]
            X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype(np.float32)
            Y = np.array(Y)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    del X, Y

    # The test batch contains exactly 1000 randomly selected images from each class.
    with open(os.path.join(ROOT, "test_batch"), "rb") as file:
        datadict = pickle.load(file, encoding="latin1")
        X_test = datadict["data"]
        y_test = datadict["labels"]
        X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype(np.float32)
        y_test = np.array(y_test)

    # The dataset contains another file, called batches.meta. It too contains a
    # Python dictionary object. It has the following entry:
    # - label_names - a 10-element list which gives meaningful names to the
    #   numeric labels
    with open(os.path.join(ROOT, "batches.meta"), "rb") as file:
        datadict = pickle.load(file, encoding="latin1")
        classes = np.array(datadict["label_names"])

    return X_train, y_train, X_test, y_test, classes


if __name__ == "__main__":
    root = "dataset"

    _ = torchvision.datasets.CIFAR10(root, download=True)

    # The folder where the CIFAR-10 images will be downloaded.
    tmp = os.path.join(root, "cifar-10-batches-py")
    X_train, y_train, X_test, y_test, classes = load_data(tmp)

    with open(os.path.join(root, "cifar10.pkl"), "wb") as f:
        pickle.dump({
            "X_train": X_train,
            "y_train": y_train,
            "X_test" : X_test,
            "y_test" : y_test,
            "classes": classes,
        }, f)

    # Remove the downloaded files.
    shutil.rmtree(tmp)
    os.remove(os.path.join(root, "cifar-10-python.tar.gz"))

#