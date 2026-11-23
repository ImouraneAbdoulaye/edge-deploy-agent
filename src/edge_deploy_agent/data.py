"""Small image-classification datasets used to train and evaluate reference models."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_LOADERS = {
    "mnist": keras.datasets.mnist.load_data,
    "fashion_mnist": keras.datasets.fashion_mnist.load_data,
}


def load_dataset(name: str) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return ((x_train, y_train), (x_test, y_test)) as float32 in [0, 1], shape (N, 28, 28, 1).

    The normalized arrays are cached in ``data/<name>.npz`` after the first download.
    """
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {sorted(_LOADERS)}")

    cache = DATA_DIR / f"{name}.npz"
    if cache.exists():
        d = np.load(cache)
        return (d["x_train"], d["y_train"]), (d["x_test"], d["y_test"])

    (x_train, y_train), (x_test, y_test) = _LOADERS[name]()
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return (x_train, y_train), (x_test, y_test)
