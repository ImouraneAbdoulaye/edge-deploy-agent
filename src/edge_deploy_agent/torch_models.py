"""PyTorch versions of the reference models, for the Qualia-CodeGen backend.

The architectures mirror ``scripts/make_reference_models.py`` (Keras) so that both
backends can be compared on the same networks. Qualia-CodeGen reads
``model.input_shape`` (channels last, without batch) to trace the graph.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn


class _Net(nn.Module):
    input_shape = (28, 28, 1)  # channels last (H, W, C), as Qualia-CodeGen expects

    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def mnist_cnn_small() -> nn.Module:
    return _Net(
        [
            nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 5 * 5, 10),
        ]
    )  # fmt: skip


def mnist_cnn_medium() -> nn.Module:
    return _Net(
        [
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 5 * 5, 128), nn.ReLU(), nn.Linear(128, 10),
        ]
    )  # fmt: skip


def fashion_mlp_large() -> nn.Module:
    return _Net(
        [
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 10),
        ]
    )  # fmt: skip


TORCH_MODELS: dict[str, Callable[[], nn.Module]] = {
    "mnist_cnn_small": mnist_cnn_small,
    "mnist_cnn_medium": mnist_cnn_medium,
    "fashion_mlp_large": fashion_mlp_large,
}
