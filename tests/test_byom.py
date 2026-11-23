"""Bring-your-own-model: registration, backend restriction, 1D models end to end."""

import shutil

import numpy as np
import pytest
import torch
from torch import nn

from edge_deploy_agent.agent import scenario as scenario_mod
from edge_deploy_agent.tools.deploy import get_model, register_model

needs_toolchain = pytest.mark.skipif(
    not all(shutil.which(t) for t in ("gcc", "g++", "cmake", "ninja", "nm")), reason="needs a C/C++ toolchain"
)


@pytest.fixture(scope="module")
def user_files(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """A tiny 1D CNN saved as a full PyTorch model, with an .npz test set."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 32, 2)).astype(np.float32)  # (N, L, C) channels last
    y = (x[:, :, 0].mean(axis=1) > 0).astype(np.int64)
    model = nn.Sequential(nn.Conv1d(2, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 30, 2))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    xt = torch.from_numpy(np.moveaxis(x, -1, 1))
    for _ in range(100):
        opt.zero_grad()
        nn.functional.cross_entropy(model(xt), torch.from_numpy(y)).backward()
        opt.step()
    model.eval()  # no input_shape attribute on purpose: it must be inferred from the data

    d = tmp_path_factory.mktemp("byom")
    torch.save(model, d / "user_cnn.pt")
    np.savez(d / "data.npz", x_test=x, y_test=y)
    return {"pt": d / "user_cnn.pt", "npz": d / "data.npz"}


@pytest.fixture
def user_scenario(user_files: dict, monkeypatch: pytest.MonkeyPatch, tmp_path) -> scenario_mod.Scenario:
    monkeypatch.setattr(scenario_mod, "DEPLOY_CACHE", tmp_path / "cache.json")
    register_model("user_cnn", user_files["npz"], torch_path=user_files["pt"])
    return scenario_mod.Scenario("user_cnn", "small")


def test_register_requires_a_model_and_valid_data(user_files: dict, tmp_path) -> None:
    with pytest.raises(ValueError, match="give a .keras"):
        register_model("nothing", user_files["npz"])
    np.savez(tmp_path / "bad.npz", x_test=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="y_test"):
        register_model("bad", tmp_path / "bad.npz", torch_path=user_files["pt"])


@needs_toolchain
def test_only_available_backends_are_offered(user_scenario: scenario_mod.Scenario) -> None:
    assert get_model("user_cnn").backends == ["qualia"]
    assert user_scenario.backends == ["qualia"]
    assert "unavailable" in user_scenario.deploy_and_check("tflite", "full_int8")["error"]
    assert "tflite" not in scenario_mod.options_description(user_scenario.backends)


@needs_toolchain
def test_inspect_and_deploy_user_1d_model(user_scenario: scenario_mod.Scenario) -> None:
    info = user_scenario.inspect()
    assert info["params"] == (2 * 3 * 4 + 4) + (120 * 2 + 2)
    res = user_scenario.deploy_and_check("qualia", "int8")
    assert res["flash_kb"] == round(info["params"] / 1024, 1)
    assert res["accuracy"] >= user_scenario.reference["qualia"] - 0.05
