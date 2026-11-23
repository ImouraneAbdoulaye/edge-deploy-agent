"""Qualia-CodeGen backend tests on a tiny synthetic PyTorch model."""

import shutil

import numpy as np
import pytest
import torch
from torch import nn

from edge_deploy_agent.tools import qualia_backend
from edge_deploy_agent.tools.qualia_backend import convert_qualia, evaluate_qualia

pytestmark = pytest.mark.skipif(
    not all(shutil.which(t) for t in ("gcc", "g++", "cmake", "ninja", "nm")),
    reason="needs a C/C++ toolchain (conda install -c conda-forge gcc gxx cmake ninja)",
)


class Toy(nn.Module):
    input_shape = (8, 8, 1)

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 6 * 6, 2))

    def forward(self, x):
        return self.layers(x)


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> dict:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.random((400, 8, 8, 1), dtype=np.float32)
    y = (x.mean(axis=(1, 2, 3)) > 0.5).astype("int64")

    model = Toy()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    xt, yt = torch.from_numpy(x.transpose(0, 3, 1, 2)), torch.from_numpy(y)
    for _ in range(100):
        opt.zero_grad()
        nn.functional.cross_entropy(model(xt), yt).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        torch_acc = float((model(xt).argmax(1) == yt).float().mean())

    d = tmp_path_factory.mktemp("qualia")
    path = d / "toy.pt"
    torch.save(model.state_dict(), path)
    return {"path": path, "x": x, "y": y, "out": d / "c", "torch_acc": torch_acc}


@pytest.fixture(autouse=True)
def _register_toy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(qualia_backend.TORCH_MODELS, "toy", Toy)


@pytest.mark.parametrize("mode,weight_bytes", [("float32", 4), ("int16", 2), ("int8", 1)])
def test_convert_measure_and_evaluate(toy: dict, mode: str, weight_bytes: int) -> None:
    conv = convert_qualia(toy["path"], mode, toy["out"], x_calib=toy["x"])
    params = (9 * 4 + 4) + (144 * 2 + 2)
    assert conv["flash_bytes"] == params * weight_bytes
    assert conv["ram_bytes"] > 0

    ev = evaluate_qualia(conv["c_dir"], toy["x"], toy["y"], max_samples=400)
    tolerance = 0.0 if mode == "float32" else 0.05
    assert ev["accuracy"] >= toy["torch_acc"] - tolerance - 1e-6


def test_integer_modes_require_calibration(toy: dict) -> None:
    with pytest.raises(ValueError, match="calibration"):
        convert_qualia(toy["path"], "int8", toy["out"])
