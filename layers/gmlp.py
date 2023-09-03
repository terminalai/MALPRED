from keras_core import ops, layers, activations, Sequential
from typing import Tuple
from utils.types import Int, Float, TensorLike

__all__ = ["SpatialGatingUnit", "gMLP"]


class SpatialGatingUnit(layers.Layer):
    def __init__(self, units, eps: Float = 1e-6, **kwargs):
        super(SpatialGatingUnit, self).__init__(**kwargs)
        self.norm = layers.LayerNormalization(epsilon=eps)
        self.proj = layers.Dense(units, bias_initializer="Ones")

    def call(self, inputs: TensorLike) -> TensorLike:
        u, v = ops.split(inputs, 2, axis=-1)

        v = self.norm(v)
        v = ops.transpose(v)
        v = self.proj(v)
        v = ops.transpose(v)

        return u * v


class gMLP(layers.Layer):
    def __init__(self, input_shape: Tuple[Int], dropout: Float = 0.1, eps: Float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.LayerNormalization(epsilon=eps)

        self.proj_in = Sequential([
            layers.Dense(input_shape[-1]*2, activation=activations.gelu),
            layers.Dropout(dropout)
        ])

        self.sgu = SpatialGatingUnit(input_shape[-2])

        self.proj_out = layers.Dense(input_shape[-1])

    def call(self, inputs: TensorLike) -> TensorLike:
        inputs = self.norm(inputs)
        x = self.proj_in(inputs)
        x = self.sgu(x)
        x = self.proj_out(x)
        return inputs + x
