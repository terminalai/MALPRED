from keras import layers
from typing import Callable, Union
from utils.types import TensorLike

__all__ = ["Residual"]


class Residual(layers.Layer):
    def __init__(self, layer: Union[layers.Layer, Callable[[TensorLike], TensorLike]], **kwargs):
        super().__init__(**kwargs)
        self.layer = layer

    def call(self, x: TensorLike) -> TensorLike:
        y = self.layer(x)
        return x + y
