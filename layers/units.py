from keras_core import ops

from utils.types import TensorLike


def glu(x: TensorLike, n_units=None) -> TensorLike:
    """Generalized linear unit nonlinear activation."""
    if n_units is None:
        n_units = ops.shape(x)[-1] // 2

    return x[..., :n_units] * ops.sigmoid(x[..., n_units:])

