import keras_core as keras
from keras_core import layers, ops
from layers.residual import Residual
from utils.types import TensorLike, Int, Float, Number
from typing import Tuple


class FocalModulation1D(layers.Layer):
    def __init__(self, dim: Int, focal_window: Int, focal_level: Int, focal_factor: Int = 2,
                 proj_dropout_rate: Float = 0.0, **kwargs):
        """The Focal Modulation layer includes query projection & context aggregation.

        Args:
            dim (int): Projection dimension.
            focal_window (int): Window size for focal modulation.
            focal_level (int): The current focal level.
            focal_factor (int): Factor of focal modulation.
            proj_dropout_rate (float): Rate of dropout.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.focal_level = focal_level

        # Project the input feature into a new feature space using a linear layer. Note the `units` used. We will be
        # projecting the input feature all at once and split the projection into query, context, and gates.
        self.initial_proj = layers.Dense(
            units=(2 * dim) + (focal_level + 1),
            use_bias=True,
        )

        self.focal_layers = []
        self.kernel_sizes = []

        for idx in range(self.focal_level):
            kernel_size = (focal_factor * idx) + focal_window
            depth_gelu_block = layers.Conv1D(
                filters=self.dim, kernel_size=kernel_size, activation=keras.activations.gelu,
                groups=self.dim, use_bias=False, padding="same"
            )
            self.focal_layers.append(depth_gelu_block)
            self.kernel_sizes.append(kernel_size)

        self.gap = layers.GlobalAveragePooling1D(keepdims=True)
        self.activation = keras.activations.gelu

        self.modulator_proj = layers.Conv1D(filters=dim, kernel_size=1, use_bias=True)

        self.proj = keras.Sequential([layers.Dense(units=dim), layers.Dropout(proj_dropout_rate)])

    def call(self, x: TensorLike) -> TensorLike:
        """Forward pass of the layer.

        Args:
            x: Tensor of shape (B, L, C)
        """

        # Apply the linear projection to the input feature map
        x_proj = self.initial_proj(x)

        # Split the projected x into query, context and gates
        query, context, gates = ops.split(
            x_proj, [self.dim, self.dim, self.focal_level+1], axis=-1
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
    """Combine FFN and Focal Modulation Layer.

    Args:
        dim (int): Number of input channels.
        input_resolution (Tuple[int]): Input resolution.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float): Dropout rate.
        drop_path (float): Stochastic depth rate.
        focal_level (int): Number of focal levels.
        focal_window (int): Focal window size at first focal level
    """

    def __init__(self, dim: Int, mlp_ratio: Number = 4.0, drop: Float = 0.0, focal_level: Int = 1,
                 focal_window: Int = 3, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

        self.modulation = Residual(FocalModulation1D(
            dim=self.dim, focal_window=focal_window, focal_level=focal_level, proj_dropout_rate=drop
        ))

        self.mlp = Residual(keras.Sequential([
            layers.LayerNormalization(epsilon=1e-5),
            layers.Dense(units=int(self.dim * mlp_ratio), activation=keras.activations.gelu),
            layers.Dense(units=self.dim), layers.Dropout(rate=drop),
        ]))

    def call(self, inputs: TensorLike) -> TensorLike:
        # Focal Modulation
        x = self.modulation(inputs)

        # FFN
        y = self.mlp(x)

        return y

