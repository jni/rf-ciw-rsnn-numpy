import numpy as np

def train_rfciw(image_data, labels,
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
    imsize = image_data.shape[1]
    receptive_fields = np.zeros(n_hidden, imsize)
    sidelen = np.sqrt(imsize)
    for i in range(n_hidden):
        mask = np.zeros((sidelen, sidelen), dtype=bool)
        inds = np.zeros((2, 2))
        while np.prod(inds[1] - inds[0]) < min_mask_size:
            inds = np.sort(np.random.randint(rf_border, sidelen - rf_border,
                                             size=(2, 2)), axis=0)
