from keras import ops
from keras.activations import selu
from keras import layers, Sequential, Model
from keras.optimizers import Lion, Adam, AdamW, SGD, Optimizer
from keras.losses import BinaryCrossentropy, Hinge, SquaredHinge, Loss, BinaryFocalCrossentropy
from layers.transformer import TransformerBlock

from utils.types import TensorLike, Float, Int, LossType, OptimizerType
from typing import Optional, Iterable

__all__ = ["TabTransformer", "tab_transformer"]


class TabTransformer(Model):
    def __init__(self, numerical_features: Optional[Iterable] = None, categorical_features: Optional[Iterable] = None,
                 num_categories: Optional[dict] = None, depth: Int = 2, heads: Int = 8, attn_dropout: Float = 0.2,
                 ff_dropout: Float = 0.2, mlp_hidden_factors: Optional[Iterable] = None,
                 mlp_num_factors: Optional[Iterable] = None, num_final: Int = 10):
        super(TabTransformer, self).__init__()

        if numerical_features is None:
            numerical_features = []
        if categorical_features is None:
            categorical_features = []
        if num_categories is None:
            num_categories = {}
        if mlp_hidden_factors is None:
            mlp_hidden_factors = [2, 4]
        if mlp_num_factors is None:
            mlp_num_factors = [2, 4]

        self.numerical = numerical_features
        self.categorical = categorical_features

        self.embedding_layers = [
            layers.Embedding(input_dim=num_categories[c], output_dim=32) for c in self.categorical
        ]

        num_columns = len(self.categorical)
        self.column_embedding = layers.Embedding(
            input_dim=num_columns, output_dim=32
        )
        self.column_indices = ops.arange(num_columns)

        self.transformers = [
            TransformerBlock(32, heads, 32, attn_dropout, ff_dropout) for _ in range(depth)
        ]

        self.flatten_transformer_output = layers.Flatten()

        num_units = [len(self.numerical) // f for f in mlp_num_factors]
        num_layers = [
            Sequential(
                layers.BatchNormalization(), layers.Dense(units, activation=selu), layers.Dropout(ff_dropout)
            ) for units in num_units
        ]

        self.num_mlp = Sequential([
            layers.LayerNormalization(),
            Sequential(num_layers),
            layers.Dense(num_final, activation="relu")
        ])

        mlp_input_dim = num_final + 32 * len(self.categorical)
        hidden_units = [mlp_input_dim // f for f in mlp_hidden_factors]

        self.final_mlp = Sequential([
            Sequential([
                layers.BatchNormalization(),
                layers.Dense(units, activation=selu),
                layers.Dropout(ff_dropout)
            ]) for units in hidden_units
        ])
        self.output_layer = layers.Dense(1, activation="sigmoid")

    def call(self, inputs: TensorLike) -> TensorLike:
        numerical_feature_list = [inputs[n] for n in self.numerical]
        categorical_feature_list = [emb(inputs[c]) for emb, c in zip(self.embedding_layers, self.categorical)]

        # TRACK 1: Categorical Values go into Transformer
        categorical_inputs = ops.concatenate(categorical_feature_list, axis=1)
        categorical_inputs += self.column_embedding(self.column_indices)

        for transformer in self.transformers:
            categorical_inputs = transformer(categorical_inputs)

        categorical_outputs = self.flatten_transformer_output(categorical_inputs)

        # TRACK 2: Numerical Values go into MLP
        numerical_inputs = ops.concatenate(numerical_feature_list, axis=1)
        numerical_outputs = self.num_mlp(numerical_inputs)

        mlp_inputs = ops.concatenate([categorical_outputs, numerical_outputs])
        x = self.final_mlp(mlp_inputs)
        output = self.output_layer(x)

        return output


def tab_transformer(
        numerical_features: Optional[Iterable] = None, categorical_features: Optional[Iterable] = None,
        num_categories: Optional[dict] = None, depth: Int = 2, heads: Int = 8, attn_dropout: Float = 0.2,
        ff_dropout: Float = 0.2, mlp_hidden_factors: Optional[Iterable] = None,
        mlp_num_factors: Optional[Iterable] = None, num_final: Int = 10, loss: LossType = "bce",
        optimizer: OptimizerType = "lion", learning_rate: float = 1e-4, metrics: Optional[Iterable] = None
) -> TabTransformer:
    if numerical_features is None:
        numerical_features = []
    if categorical_features is None:
        categorical_features = []
    if num_categories is None:
        num_categories = {}
    if metrics is None:
        metrics = ['accuracy', "Precision", "Recall", "AUC"]

    tt = TabTransformer(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        num_categories=num_categories,
        depth=depth, heads=heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout,
        mlp_hidden_factors=mlp_hidden_factors, mlp_num_factors=mlp_num_factors, num_final=num_final
    )

    if isinstance(optimizer, Optimizer):
        opt = optimizer
    elif optimizer == "lion":
        opt = Lion(learning_rate=learning_rate)
    elif optimizer == "adam":
        opt = Adam(learning_rate=learning_rate)
    elif optimizer.startswith("adamw_"):
        weight_decay = float(optimizer.split("_")[-1].strip())
        opt = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        opt = SGD(learning_rate=learning_rate)

    if isinstance(loss, Loss):
        ls = loss
    elif loss == "hinge":
        ls = Hinge()
    elif loss == "sqhinge":
        ls = SquaredHinge()
    elif loss == "focal":
        ls = BinaryFocalCrossentropy()
    else:
        ls = BinaryCrossentropy()

    tt.compile(optimizer=opt,
               loss=ls,
               metrics=metrics)

    return tt
