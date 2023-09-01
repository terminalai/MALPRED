# utils/training.py

import tensorflow as tf
from typing import Tuple
from keras_core import Model
from keras_core.callbacks import History

__all__ = [
    "preprocess", "read_csv",
    "training_curve"
]


def preprocess(inp: dict, out: tf.Tensor) -> Tuple[dict, tf.Tensor]:
    for key, value in inp.items():
        inp[key] = inp[key][:, tf.newaxis]
    return inp, out


def read_csv(directory: str, label_col: str = "HasDetections", batch_size=64) -> tf.data.Dataset:
    return tf.data.experimental.make_csv_dataset(
        directory,
        batch_size=batch_size,
        label_name=label_col,
        num_epochs=1,
        ignore_errors=True
    ).map(preprocess)


def training_curve(train_dir: str, test_dir: str, model: Model) -> History:
    train = read_csv(train_dir)
    test = read_csv(test_dir)

    history = model.fit(
        x=train,
        validation_data=test,
        epochs=20,
        shuffle=True
    )

    del train
    del test

    return history
