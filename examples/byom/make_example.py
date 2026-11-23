"""Example for the "bring your own model" mode: a 1D CNN classifying 3-axis sensor windows.

    python examples/byom/make_example.py
    python scripts/run_agent.py sensor_cnn STM32L476RG \
        --pt examples/byom/out/sensor_cnn.pt --keras examples/byom/out/sensor_cnn.keras \
        --data examples/byom/out/sensor_data.npz

Synthetic accelerometer-like signals (128 samples x 3 axes), 4 classes: idle (noise),
walk (slow oscillation), run (faster, stronger), shake (high-frequency bursts). The same
architecture is trained in PyTorch (saved as a full model for Qualia) and in Keras.
"""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import torch
from torch import nn

OUT = Path(__file__).parent / "out"
LENGTH, AXES, CLASSES = 128, 3, 4


def make_signals(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(LENGTH) / 50.0  # 50 Hz sampling
    x = np.empty((n, LENGTH, AXES), np.float32)
    y = rng.integers(0, CLASSES, n)
    for i, label in enumerate(y):
        phase = rng.uniform(0, 2 * np.pi, AXES)
        noise = rng.normal(0, 0.45, (LENGTH, AXES))
        if label == 0:
            sig = 0
        elif label == 1:
            sig = 0.5 * np.sin(2 * np.pi * rng.uniform(1.5, 2.8) * t[:, None] + phase)
        elif label == 2:
            sig = 0.7 * np.sin(2 * np.pi * rng.uniform(2.4, 3.6) * t[:, None] + phase)
        else:
            sig = np.sign(np.sin(2 * np.pi * rng.uniform(7, 10) * t[:, None] + phase)) * rng.uniform(0.3, 0.8)
        x[i] = sig + noise
    return x, y


def torch_model() -> nn.Module:
    model = nn.Sequential(
        nn.Conv1d(AXES, 16, 5), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(16, 32, 5), nn.ReLU(), nn.MaxPool1d(2),
        nn.Flatten(), nn.Linear(32 * 29, CLASSES),
    )  # fmt: skip
    model.input_shape = (LENGTH, AXES)  # channels last, read by Qualia-CodeGen
    return model


def keras_model() -> keras.Model:
    layers = keras.layers
    return keras.Sequential(
        [
            keras.Input((LENGTH, AXES)),
            layers.Conv1D(16, 5, activation="relu"), layers.MaxPooling1D(2),
            layers.Conv1D(32, 5, activation="relu"), layers.MaxPooling1D(2),
            layers.Flatten(), layers.Dense(CLASSES),
        ]
    )  # fmt: skip


def main() -> None:
    rng = np.random.default_rng(0)
    x_train, y_train = make_signals(4000, rng)
    x_test, y_test = make_signals(1000, rng)
    OUT.mkdir(exist_ok=True)

    torch.manual_seed(0)
    tm = torch_model()
    opt = torch.optim.Adam(tm.parameters(), lr=1e-3)
    xt, yt = torch.from_numpy(np.moveaxis(x_train, -1, 1)), torch.from_numpy(y_train)
    for _ in range(8):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            idx = perm[i : i + 64]
            opt.zero_grad()
            nn.functional.cross_entropy(tm(xt[idx]), yt[idx]).backward()
            opt.step()
    tm.eval()
    with torch.no_grad():
        acc = (tm(torch.from_numpy(np.moveaxis(x_test, -1, 1))).argmax(1).numpy() == y_test).mean()
    torch.save(tm, OUT / "sensor_cnn.pt")  # full model, not a state_dict
    print(f"[torch] test accuracy {acc:.3f}")

    km = keras_model()
    km.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    km.fit(x_train, y_train, epochs=8, batch_size=64, verbose=0)
    print(f"[keras] test accuracy {km.evaluate(x_test, y_test, verbose=0)[1]:.3f}")
    km.save(OUT / "sensor_cnn.keras")

    np.savez_compressed(OUT / "sensor_data.npz", x_calib=x_train[:1000], x_test=x_test, y_test=y_test)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
