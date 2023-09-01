from keras_core import ops


def glu(x, n_units=None):
    """Generalized linear unit nonlinear activation."""
    if n_units is None:
        n_units = ops.shape(x)[-1] // 2

    return x[..., :n_units] * ops.sigmoid(x[..., n_units:])

