# layers/scarf.py

import keras_core as keras
from keras_core import layers, ops, initializers, activations
import tensorflow as tf

from utils.types import Int, Float, TensorLike
from typing import Optional

__all__ = ["SCARF"]


def _scarf_dense(dim: Int) -> layers.Dense:
    return layers.Dense(
        dim, activation=activations.relu,
        bias_initializer=initializers.Constant(0.01)
    )


class _SCARFMLP(layers.Layer):
    def __init__(self, hidden_dim: Int, depth: Int, dropout: Float = 0.0, **kwargs):
        super().__init__(**kwargs)
        encoder_layers = []

        for i in range(depth):
            encoder_layers.append(_scarf_dense(hidden_dim))
            if i + 1 == depth:
                encoder_layers.append(layers.Dropout(dropout))

        self.mlp = keras.Sequential(encoder_layers)

    def call(self, x: TensorLike) -> TensorLike:
        return self.mlp(x)


class SCARF(layers.Layer):
    def __init__(
            self,
            input_dim: Int,
            hidden_dim: Int,
            encoder_depth: Int = 4,
            head_depth: Int = 2,
            corruption_rate: Float = 0.6,
            encoder: Optional[layers.Layer] = None,
            pretraining_head: Optional[layers.Layer] = None,
            dropout: Float = 0.0,
            **kwargs
    ):
        """Implementation of SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption.

        SCARF implements an encoder that learns the embeddings by minimizing the contrastive loss of a sample and a
        corrupted view of it. This corrupted view is built by replacing a random set of features by another sample
        randomly drawn independently.

        Args:
            input_dim (int): size of the inputs
            hidden_dim (int): dimension of the embedding space
            encoder_depth (int, optional): number of layers of the encoder MLP. Defaults to 4.
            head_depth (int, optional): number of layers of the pretraining head. Defaults to 2.
            corruption_rate (float, optional): fraction of features to corrupt. Defaults to 0.6.
            encoder (nn.Module, optional): encoder network to build the embeddings. Defaults to None.
            pretraining_head (nn.Module, optional): pretraining head for the training procedure. Defaults to None.
            dropout (float, optional): probability of dropout. Defaults to 0.0.
        """
        super().__init__(**kwargs)

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = _SCARFMLP(hidden_dim, encoder_depth, dropout)

        if pretraining_head:
            self.pretraining_head = pretraining_head
        else:
            self.pretraining_head = _SCARFMLP(hidden_dim, head_depth, dropout)

        self.corruption_len = int(corruption_rate * input_dim)

    def call(self, anchor: TensorLike, random_sample: TensorLike) -> TensorLike:
        b, m = ops.shape(anchor)

        # 1: create a mask of size (batch size, m) where for each sample we set the
        # jth column to True at random, such that corruption_len / m = corruption_rate
        # 3: replace x_1_ij by x_2_ij where mask_ij is true to build x_corrupted

        corruption_mask = ops.zeros_like(anchor, dtype=bool)

        for i in range(b):
            corruption_idx = tf.random.shuffle(ops.arange(m))[:self.corruption_len]
            corruption_mask[i, corruption_idx] = True

        positive = ops.where(corruption_mask, random_sample, anchor)

        emb_anchor = self.encoder(anchor)
        emb_anchor = self.pretraining_head(emb_anchor)

        emb_positive = self.encoder(positive)
        emb_positive = self.pretraining_head(emb_positive)

        return emb_anchor, emb_positive
