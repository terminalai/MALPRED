from keras_core.activations import gelu
from keras_core import layers, Sequential

__all__ = ["TransformerBlock"]


class TransformerBlock(layers.Layer):
    def __init__(
            self, embed_dim: int, num_heads: int, ff_dim: int,
            att_dropout: float = 0.1, ff_dropout: float = 0.1
    ):
        super().__init__()

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=att_dropout)
        self.skip1 = layers.Add()

        self.ffn = Sequential([
            layers.Dense(ff_dim, activation=gelu),
            layers.Dropout(ff_dropout),
            layers.Dense(embed_dim),
        ])
        self.skip2 = layers.Add()
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        inputs = self.layernorm1(inputs)
        attention_output = self.att(inputs, inputs)
        attention_output = self.skip1([inputs, attention_output])
        feedforward_output = self.ffn(attention_output)
        transformer_output = self.skip2([feedforward_output, attention_output])
        transformer_output = self.layernorm2(transformer_output)
        return transformer_output
