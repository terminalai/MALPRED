import keras_core as keras
from keras_core import layers, ops, activations
from layers.residual import Residual
from typing import Tuple
from utils.types import Float, TensorLike

__all__ = ["FNetLayer"]


def _fnet_fft(inputs: TensorLike) -> TensorLike:
    return ops.fft2((inputs, ops.zeros_like(inputs)))[0]


class FNetLayer(layers.Layer):
    def __init__(self, dropout_rate: Float, eps: Float, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.fft = Residual(_fnet_fft)
        self.norm = layers.LayerNormalization(epsilon=eps)
        self.ffn = lambda x: x

    def build(self, input_shape: Tuple):
        embedding_dim = input_shape[-1]
        self.ffn = Residual(keras.Sequential([
            layers.Dense(units=embedding_dim, activation=activations.gelu),
            layers.Dropout(rate=self.dropout_rate),
            layers.Dense(units=embedding_dim),
        ]))

    def call(self, inputs: TensorLike) -> TensorLike:
        # Apply fourier transformations.
        x = self.fft(inputs)
        # Apply layer normalization.
        x = self.norm(x)
        # Apply Feedforward network.
        x = self.ffn(x)
        # Apply layer normalization.
        return self.norm(x)
