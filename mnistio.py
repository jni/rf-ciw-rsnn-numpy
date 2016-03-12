import os
import numpy as np

# this dictionary is derived from the documentation in this page:
# http://yann.lecun.com/exdb/mnist/
# The third byte codes the type of the data:
# 0x08: unsigned byte
# 0x09: signed byte
# 0x0B: short (2 bytes)
# 0x0C: int (4 bytes)
# 0x0D: float (4 bytes)
# 0x0E: double (8 bytes)
dtype_dict = {
    8: np.uint8,
    9: np.int8,
    11: np.int16,
    12: np.int32,
    13: np.float32,
    14: np.float64
}


def read_mnist_header(filename):
    """Helper function to flip the byte endianness of the header in IDX format.

    Parameters
    ----------
    filename : string
        The name of the file being read.

    Returns
    -------
    dtype : numpy.dtype
        The datatype of the data in the file
    ndim : int
        The number of dimensions of the data
    shape : tuple of int
        The shape of the data
    """
    _, _, dtype_code, ndim = np.memmap(filename, mode='r', shape=(4,))
    dtype = dtype_dict[dtype_code]
    shape = tuple(np.memmap(filename, offset=4,
                            dtype=np.int32, shape=(ndim,)).byteswap())
    return dtype, ndim, shape


def read_mnist_data(filename, array=True):
    """Read the data array from the filename.

    Parameters
    ----------
    filename : string
        The file to read.
    array : bool, optional
        If `True`, an array is returned, otherwise, the original memory
        map is returned.

    Returns
    -------
    arr : numpy array or memmap
    """
    dtype, ndim, shape = read_mnist_header(filename)
    memmap = np.memmap(filename, offset=4 * (ndim + 1), dtype=dtype,
                       shape=shape, mode='r').byteswap()
    if array:
        return np.array(memmap)
    else:
        return memmap


def mnist(directory, ravel_images=True, normalise_images=True):
    """Return train and test datasets assuming standard names in a directory.

    Parameters
    ----------
    directory : string
        Path to uncompressed mnist data.
    ravel_images : bool, optional
        Return the images as linear lists of pixels, rather than 2D arrays.
    normalise_images : bool, optional
        Convert uint8 data in [0, 255] to float data in [0, 1]. Note that
        the parameters determined in McDonnell et al 2015 [1]_ work for
        normalised data.

    Returns
    -------
    Xtr, ytr, Xts, yts : uint8 arrays, shape (nsamples, npixels) or (nsamples,)
        The training images, training labels, testing images, and testing
        labels.

    References
    ----------
    .. [1] http://arxiv.org/abs/1412.8307
    """
    names = ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
             't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
    data = [read_mnist_data(os.path.join(directory, n)) for n in names]
    if ravel_images:
        data[0] = np.reshape(data[0], (data[0].shape[0], -1))
        data[2] = np.reshape(data[2], (data[2].shape[0], -1))
    if normalise_images:
        data[0] = data[0] / 255
        data[2] = data[2] / 255
    return data
