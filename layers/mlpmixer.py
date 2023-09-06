import keras_core as keras
from keras_core import ops, layers, activations

from layers.residual import Residual
from typing import Tuple
from utils.types import Int, Float, TensorLike

__all__ = ["MLPMixerBlock"]


class _MLPSpatialMixer(layers.Layer):
    def __init__(self, spatial_dim: Int, eps: Float = 1e-6, **kwargs):
        super().__init__(**kwargs)

        self.norm = layers.LayerNormalization(epsilon=eps)

        self.mlp = keras.Sequential([
            layers.Dense(units=spatial_dim, activation=activations.gelu),
            layers.Dense(units=spatial_dim, activation=activations.gelu)
        ])

    def call(self, inputs: TensorLike) -> TensorLike:
        x = self.norm(inputs)
        x = ops.transpose(x, (0, 2, 1))
        x = self.mlp(x)
        x = ops.transpose(x, (0, 2, 1))
        return x


class MLPMixerBlock(layers.Layer):
    def __init__(self, dropout_rate: Float, eps: Float, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.eps = eps

        self.spatial_mixer = lambda x: x
        self.channel_mixer = lambda x: x

    def build(self, input_shape: Tuple):
        spatial_dim = input_shape[-2]
        embedding_dim = input_shape[-1]

        self.spatial_mixer = Residual(_MLPSpatialMixer(spatial_dim, eps=self.eps))

        self.channel_mixer = Residual(keras.Sequential([
            layers.LayerNormalization(epsilon=self.eps),
            layers.Dense(units=spatial_dim, activation=activations.gelu),
            layers.Dense(units=embedding_dim),
            layers.Dropout(rate=self.dropout_rate),
        ]))

    def call(self, inputs: TensorLike) -> TensorLike:
        x = self.spatial_mixer(inputs)
        y = self.channel_mixer(x)
        return y
