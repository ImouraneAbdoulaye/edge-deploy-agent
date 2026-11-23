"""Train the reference models used as deployment scenarios, in Keras and in PyTorch.

    python scripts/make_reference_models.py

Each model is deliberately sized so that it fits some targets as-is, others only
after quantization, and some not at all.
"""

from __future__ import annotations

from pathlib import Path

import keras
import torch
from keras import layers

from edge_deploy_agent.data import load_dataset
from edge_deploy_agent.torch_models import TORCH_MODELS

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def mnist_cnn_small() -> keras.Model:
    return keras.Sequential(
        [
            keras.Input((28, 28, 1)),
            layers.Conv2D(8, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(16, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(10),
        ],
        name="mnist_cnn_small",
    )


def mnist_cnn_medium() -> keras.Model:
    return keras.Sequential(
        [
            keras.Input((28, 28, 1)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10),
        ],
        name="mnist_cnn_medium",
    )


def fashion_mlp_large() -> keras.Model:
    return keras.Sequential(
        [
            keras.Input((28, 28, 1)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(10),
        ],
        name="fashion_mlp_large",
    )


REFERENCE_MODELS = {
    "mnist_cnn_small": (mnist_cnn_small, "mnist"),
    "mnist_cnn_medium": (mnist_cnn_medium, "mnist"),
    "fashion_mlp_large": (fashion_mlp_large, "fashion_mnist"),
}


def train_keras(name: str, build, x_train, y_train, x_test, y_test, epochs: int) -> None:
    model = build()
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_split=0.1, verbose=2)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    path = MODELS_DIR / f"{name}.keras"
    model.save(path)
    print(f"[keras] {name}: {model.count_params():,} params, test accuracy {acc:.4f} -> {path}")


def train_torch(name: str, x_train, y_train, x_test, y_test, epochs: int) -> None:
    torch.manual_seed(0)
    model = TORCH_MODELS[name]()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Keras layout (N, H, W, C) -> PyTorch layout (N, C, H, W)
    x = torch.from_numpy(x_train.transpose(0, 3, 1, 2))
    y = torch.from_numpy(y_train).long()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 128):
            idx = perm[i : i + 128]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
        print(f"[torch] {name} epoch {epoch + 1}/{epochs} loss {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(x_test.transpose(0, 3, 1, 2))).argmax(1).numpy()
    path = MODELS_DIR / f"{name}.pt"
    torch.save(model.state_dict(), path)
    params = sum(p.numel() for p in model.parameters())
    print(f"[torch] {name}: {params:,} params, test accuracy {(preds == y_test).mean():.4f} -> {path}")


def main(epochs: int = 3) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    for name, (build, dataset) in REFERENCE_MODELS.items():
        (x_train, y_train), (x_test, y_test) = load_dataset(dataset)
        train_keras(name, build, x_train, y_train, x_test, y_test, epochs)
        train_torch(name, x_train, y_train, x_test, y_test, epochs)


if __name__ == "__main__":
    main()
