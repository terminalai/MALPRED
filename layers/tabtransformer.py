from keras_core import ops
from keras_core.activations import selu, gelu
from keras_core import layers, Sequential, Model
from keras_core.optimizers import Lion, Adam
from keras_core.losses import BinaryCrossentropy


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


class TabTransformer(Model):
    def __init__(
            self,
            numerical_features=None,
            categorical_features=None,
            num_categories=None,
            depth: int = 2,
            heads: int = 8,
            attn_dropout: float = 0.2,
            ff_dropout: float = 0.2,
            mlp_hidden_factors=None,
            mlp_num_factors=None,
            num_final: int = 10,
    ):
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

        self.continuous_normalization = layers.LayerNormalization()
        self.numerical_concatenation = layers.Concatenate(axis=1)  # 1

        self.cat_embedding_layers = [
            layers.Embedding(
                input_dim=num_categories[c], output_dim=32
            ) for c in self.categorical
        ]

        num_columns = len(self.categorical)

        self.column_embedding = layers.Embedding(
            input_dim=num_columns, output_dim=32
        )
        self.column_indices = ops.arange(num_columns)

        self.embedded_concatenation = layers.Concatenate(axis=1)

        self.transformers = []
        for _ in range(depth):
            self.transformers.append(
                TransformerBlock(
                    32, heads, 32,
                    attn_dropout, ff_dropout,
                )
            )
        self.flatten_transformer_output = layers.Flatten()
        self.pre_mlp_concatenation = layers.Concatenate()

        numerical_dim = len(self.numerical)
        num_units = [numerical_dim // f for f in mlp_num_factors]
        num_layers = []
        for units in num_units:
            num_layers.append(layers.BatchNormalization()),
            num_layers.append(layers.Dense(units, activation=selu))
            num_layers.append(layers.Dropout(ff_dropout))

        self.num_final = Sequential(num_layers)
        self.num_layer = layers.Dense(num_final, activation="relu")

        mlp_input_dim = num_final + 32 * len(self.categorical)

        hidden_units = [mlp_input_dim // f for f in mlp_hidden_factors]
        mlp_layers = []
        for units in hidden_units:
            mlp_layers.append(layers.BatchNormalization()),
            mlp_layers.append(layers.Dense(units, activation=selu))
            mlp_layers.append(layers.Dropout(ff_dropout))

        self.mlp_final = Sequential(mlp_layers)
        self.output_layer = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        numerical_feature_list = [inputs[n] for n in self.numerical]
        categorical_feature_list = [emb(inputs[c]) for emb, c in zip(self.cat_embedding_layers, self.categorical)]

        transformer_inputs = self.embedded_concatenation(categorical_feature_list)

        # Add column embeddings
        transformer_inputs += self.column_embedding(self.column_indices)

        for transformer in self.transformers:
            transformer_inputs = transformer(transformer_inputs)

        mlp_input = self.flatten_transformer_output(transformer_inputs)

        numerical_inputs = self.numerical_concatenation(numerical_feature_list)
        numerical_inputs = self.continuous_normalization(numerical_inputs)
        numerical_inputs = self.num_final(numerical_inputs)
        numerical_inputs = self.num_layer(numerical_inputs)
        mlp_input = self.pre_mlp_concatenation([mlp_input, numerical_inputs])
        x = self.mlp_final(mlp_input)
        output = self.output_layer(x)
        return output


def tab_transformer(
        numerical_features=None, categorical_features=None, num_categories=None,
        depth: int = 2, heads: int = 8, attn_dropout: float = 0.2, ff_dropout: float = 0.2,
        mlp_hidden_factors=None, mlp_num_factors=None, num_final: int = 10,
        loss="bce", optimizer="lion", learning_rate=1e-4, metrics=None
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

    if optimizer == "lion":
        opt = Lion(learning_rate=learning_rate)
    elif optimizer == "adam":
        opt = Adam(learning_rate=learning_rate)
    else:
        opt = Adam(learning_rate=learning_rate)

    if loss == "bce":
        ls = BinaryCrossentropy()
    else:
        ls = BinaryCrossentropy()

    tt.compile(optimizer=opt,
               loss=ls,
               metrics=metrics)

    return tt
