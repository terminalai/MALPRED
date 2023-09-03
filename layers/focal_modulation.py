from keras_core import layers, ops, activations, Sequential
from layers.residual import Residual

from utils.types import Int, Float, TensorLike

__all__ = ["FocalModulation", "FocalModulationBlock"]


class FocalModulation(layers.Layer):
    def __init__(self, embed_dim: Int, focal_window: Int, focal_level: Int, focal_factor: Int = 2,
                 proj_dropout_rate: Float = 0.0, **kwargs):
        """The Focal Modulation layer includes query projection & context aggregation.

        This has been reimplemented from https://keras.io/examples/vision/focal_modulation_network/
        to deal with 1D data.

        Args:
            embed_dim (int): Projection dimension.
            focal_window (int): Window size for focal modulation.
            focal_level (int): The current focal level.
            focal_factor (int): Factor of focal modulation.
            proj_dropout_rate (float): Rate of dropout.
        """
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.focal_level = focal_level

        # Project the input feature into a new feature space using a linear layer. Note the `units` used. We will be
        # projecting the input feature all at once and split the projection into query, context, and gates.
        self.initial_proj = layers.Dense(
            units=(2 * embed_dim) + (focal_level + 1),
            use_bias=True,
        )

        self.focal_layers = []
        self.kernel_sizes = []

        for idx in range(self.focal_level):
            kernel_size = (focal_factor * idx) + focal_window
            depth_gelu_block = layers.Conv1D(
                filters=embed_dim, kernel_size=kernel_size, activation=activations.gelu,
                groups=embed_dim, use_bias=False, padding="same"
            )
            self.focal_layers.append(depth_gelu_block)
            self.kernel_sizes.append(kernel_size)

        self.gap = layers.GlobalAveragePooling1D(keepdims=True)
        self.activation = activations.gelu

        self.modulator_proj = layers.Conv1D(filters=embed_dim, kernel_size=1, use_bias=True)

        self.proj = Sequential([layers.Dense(units=embed_dim), layers.Dropout(proj_dropout_rate)])

    def call(self, x: TensorLike) -> TensorLike:
        """Forward pass of the layer.

        Args:
            x: Tensor of shape (B, L, C)
        """

        # Apply the linear projection to the input feature map
        x_proj = self.initial_proj(x)

        # Split the projected x into query, context and gates
        query, context, gates = ops.split(
            x_proj, [self.embed_dim, 2*self.embed_dim], axis=-1
        )

        # Context aggregation
        context_all = ops.zeros_like(context)

        for idx in range(self.focal_level):
            context = self.focal_layers[idx](context)
            context_all += context * gates[..., idx:idx+1]

        # Build the global context
        context_global = self.activation(self.gap(context))
        context_all += context_global * gates[..., self.focal_level:]

        # Focal Modulation
        modulator = self.modulator_proj(context_all)
        x_output = query * modulator

        # Project the output and apply dropout
        x_output = self.proj(x_output)

        return x_output


class FocalModulationBlock(layers.Layer):
    def __init__(self, embed_dim: Int, focal_level: Int, focal_window: Int,
                 ff_dim: Int, focal_dropout: Float = 0.0, ff_dropout: Float = 0.0,
                 eps: Float = 1e-6, **kwargs):
        super().__init__(**kwargs)

        self.mod = Residual(FocalModulation(
                embed_dim=embed_dim, focal_window=focal_window, focal_level=focal_level, proj_dropout_rate=focal_dropout
        ))

        self.modnorm = layers.LayerNormalization(epsilon=eps)

        self.ffn = Residual(Sequential([
            layers.Dense(ff_dim, activation=activations.gelu),
            layers.Dropout(rate=ff_dropout),
            layers.Dense(embed_dim)
        ]))

        self.ffnnorm = layers.LayerNormalization(epsilon=eps)

    def call(self, inputs: TensorLike) -> TensorLike:
        # Focal Modulation
        x = self.modnorm(self.mod(inputs))

        # FFN
        y = self.ffnnorm(self.ffn(x))

        return y
