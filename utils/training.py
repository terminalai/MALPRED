# utils/training.py

import tensorflow as tf
from typing import Tuple
from keras import Model
from keras.callbacks import History
from utils.types import TensorLike
from utils.constants import NUM_CATEGORIES

__all__ = [
    "preprocess", "read_csv",
    "training_curve"
]


def preprocess(inp: dict, out: TensorLike) -> Tuple[dict, TensorLike]:
    for key, value in inp.items():
        inp[key] = inp[key][:, tf.newaxis]
    return inp, out

def one_hot(inp: dict, out: TensorLike) -> Tuple[dict, TensorLike]:
    for key, value in inp.items():
        if  key in NUM_CATEGORIES:
            inp[key] = tf.one_hot(inp[key], NUM_CATEGORIES[key])
    return inp, out

def read_csv(directory: str, label_col: str = "HasDetections", batch_size=64) -> tf.data.Dataset:
    return tf.data.experimental.make_csv_dataset(
        directory,
        batch_size=batch_size,
        label_name=label_col,
        num_epochs=1,
        ignore_errors=True
    ).map(preprocess)


def training_curve(train_dir: str, test_dir: str, model: Model, onehot=False) -> History:
    train = read_csv(train_dir)
    test = read_csv(test_dir)

    if onehot:
        train = train.map(one_hot)
        test = test.map(one_hot)

    history = model.fit(
        x=train,
        validation_data=test,
        epochs=20,
        shuffle=True
    )

    del train
    del test

    return history
