import numpy as np
from skimage import segmentation as seg

eps = np.finfo(float).eps


def _generate_receptive_fields(n_hidden, image_side_length,
                               min_mask_size, rf_border):
    """Generate the random receptive fields.

    Parameters
    ----------
    n_hidden : int
        The number of hidden neurons.
    image_side_length : int
        The side length of an image
    min_mask_size : int
        The minimum area covered by a random receptive field.
    rf_border : int
        Don't set up receptive fields less than this many pixels away
        from the image border.

    Returns
    -------
    receptive_fields : array of float, shape (n_hidden, image_side_length ** 2)
        The receptive fields.
    """
    sidelen = image_side_length
    imsize = sidelen * sidelen
    receptive_fields = np.zeros((n_hidden, imsize), dtype=bool)
    for i in range(n_hidden):
        mask = np.reshape(receptive_fields[i], (sidelen, sidelen))
        inds = np.zeros((2, 2))
        while np.prod(inds[1] - inds[0]) < min_mask_size:
            inds = np.sort(np.random.randint(rf_border, sidelen - rf_border,
                                             size=(2, 2)), axis=0)
        rows, cols = inds.T
        mask[rows[0]:rows[1], cols[0]:cols[1]] = True
    return receptive_fields


def _generate_random_weights(images, labels, receptive_fields,
                             input_weight_scaling):
    """Generate the random input neuron weights.

    Parameters
    ----------
    images : array, shape (n_images, image_size)
        The input images.
    labels : array of int, shape (n_images)
        The image labels.
    receptive_fields : array of bool, shape (n_hidden, image_size)
        The neuron receptive fields (the locations of pixels in the
        image that each hidden neuron is sensitive to.
    input_weight_scaling : float
        Scale the random weights by this factor, before adding the bias.

    Returns
    -------
    w_random : array of float, shape (n_hidden, imsize + 1)
        The random weights, including an extra column to implement bias.
    """
    n_hidden, imsize = receptive_fields.shape
    nimages = labels.shape[0]
    biases = np.zeros((n_hidden, 1), dtype=float)
    w_random = np.zeros((n_hidden, imsize), dtype=float)
    for i in range(n_hidden):
        norm, n1, n2 = 0, 0, 0
        while labels[n1] == labels[n2] or norm < eps:
            n1, n2 = np.random.choice(nimages, size=2, replace=False)
            wrow = receptive_fields[i] * (images[n1] - images[n2])
            norm = np.linalg.norm(wrow)
        w_random[i] = wrow / norm
        biases[i] = 0.5 * (images[n1] + images[n2]) @ w_random[i]
    w_random *= input_weight_scaling
    w_random = np.concatenate((w_random, biases), axis=1)
    return w_random


def one_hot(labels):
    """One-hot encode sequential labels {0, 1, ..., m}.

    Parameters
    ----------
    labels : array of int, shape (n,)
        The input labels, encoded as sequential integers.

    Returns
    -------
    encoded : array of int in {0, 1}, shape (n, m + 1)
        The one-hot encoding of the input labels.
    """
    n = len(labels)
    m = np.max(labels)
    encoded = np.zeros((n, m + 1))
    encoded[np.arange(n), labels] = 1
    return encoded


def train_rfciw(images, labels,
                n_hidden=1600, input_weight_scaling=2, ridge=1e-8,
                min_mask_size=10, rf_border=3):
    """Train the neural network, given input images and target labels.

    Parameters
    ----------
    images : array, shape (n_images, image_size)
        The input image data, with raveled (linearised) images.
    labels : array, shape (n_images,)
        The image labels.
    n_hidden : int, optional
        The number of hidden neurons. In general, a higher number gives
        better prediction accuracy.
    input_weight_scaling : float, optional
        Rescale the normalised input weights by this factor.
    ridge : float, optional
        The regularising ridge regression factor.
    min_mask_size : int, optional
        The minimum area covered by a random receptive field.
    rf_border : int, optional
        Don't set up receptive fields less than this many pixels away
        from the image border.

    Returns
    -------
    w_random : array, shape (n_hidden, image_size + 1)
        The random receptive field weights. Bias is implemented by an
        additional column in this matrix, corresponding to an appended
        column in the image data being predicted.
    w_out : array, shape (n_hidden, n_classes)
        The trained output weights corresponding to the input receptive
        fields.
    """
    nimages = images.shape[0]
    imsize = images.shape[1]
    sidelen = np.sqrt(imsize)

    # generate the input weights matrix
    receptive_fields = _generate_receptive_fields(n_hidden, sidelen,
                                                  min_mask_size, rf_border)
    w_random = _generate_random_weights(images, labels, receptive_fields,
                                        input_weight_scaling)

    # add bias column to the image data
    images = np.concatenate((images, np.ones((images.shape[0], 1))), axis=1)

    activations = 1 / (1 + np.exp(-w_random @ images.T))

    # generate targets matrix from labels (one-hot encoding)
    targets = one_hot(labels)

    # solve for the output weights
    w_out = np.linalg.lstsq(activations @ activations.T +
                            ridge * np.ones((n_hidden, n_hidden)),
                            activations @ targets)[0]
    return w_random, w_out


def predict_class(images, w_random, w_out):
    images = np.concatenate((images, np.ones((images.shape[0], 1))), axis=1)
    prediction_matrix = w_out.T @ (1 / (1 + np.exp(-w_random @ images.T)))
    predicted_labels = np.argmax(prediction_matrix, axis=0)
    return predicted_labels


def test_accuracy(images, labels, w_random, w_out):
    predicted_labels = predict_class(images, w_random, w_out)
    return np.mean(labels == predicted_labels)


if __name__ == '__main__':
    import mnistio
    Xtr, ytr, Xts, yts = mnistio.mnist('data')
    w_random, w_out = train_rfciw(Xtr, ytr)
    yts_pred = predict_class(Xts, w_random, w_out)
    print('accuracy: %.3f' % np.mean((yts_pred == yts).astype(float)))
