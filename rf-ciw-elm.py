import numpy as np

eps = np.finfo(float).eps

def train_rfciw(images, labels,
                n_hidden=1600, input_weight_scaling=2, ridge=1e-8,
                min_mask_size=10, rf_border=3):
    """Train the neural network, given input images and target labels.

    Parameters
    ----------
    images
    labels

    rf_border : int, optional
        Don't set up receptive fields less than this many pixels away
        from the image border.

    Returns
    -------

    """

    nimages = images.shape[0]
    imsize = images.shape[1]
    sidelen = np.sqrt(imsize)
    # generate random receptive fields
    receptive_fields = np.zeros(n_hidden, imsize, dtype=bool)
    for i in range(n_hidden):
        mask = np.reshape(receptive_fields[i], (sidelen, sidelen))
        inds = np.zeros((2, 2))
        while np.prod(inds[1] - inds[0]) < min_mask_size:
            inds = np.sort(np.random.randint(rf_border, sidelen - rf_border,
                                             size=(2, 2)), axis=0)
        rows, cols = inds.T
        mask[rows[0]:rows[1], cols[0]:cols[1]] = True

    # generate constrained weights
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
