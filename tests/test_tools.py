"""Tool tests on a tiny synthetic model (no dataset download needed)."""

from pathlib import Path

import keras
import numpy as np
import pytest

from edge_deploy_agent.tools.convert import QUANT_MODES, convert_model
from edge_deploy_agent.tools.evaluate import check_constraints, evaluate_tflite
from edge_deploy_agent.tools.inspect_model import inspect_model


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """A small conv net trained on a trivially separable problem."""
    rng = np.random.default_rng(0)
    x = rng.random((400, 8, 8, 1), dtype=np.float32)
    y = (x.mean(axis=(1, 2, 3)) > 0.5).astype("int64")

    model = keras.Sequential(
        [
            keras.Input((8, 8, 1)),
            keras.layers.Conv2D(4, 3, activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(2),
        ]
    )
    model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(x, y, epochs=5, verbose=0)

    path = tmp_path_factory.mktemp("model") / "toy.keras"
    model.save(path)
    return {"path": path, "x": x, "y": y, "out": path.parent / "tflite"}


def test_inspect(toy: dict) -> None:
    info = inspect_model(toy["path"])
    assert info["input_shape"] == [8, 8, 1]
    assert info["params"] == (3 * 3 * 4 + 4) + (6 * 6 * 4 * 2 + 2)
    assert info["float32_weights_bytes"] == info["params"] * 4
    # Conv layer: 64 input elements + 6*6*4 = 144 output elements alive together.
    assert info["peak_activation_elements"] == 64 + 144


@pytest.mark.parametrize("mode", list(QUANT_MODES))
def test_convert_and_evaluate(toy: dict, mode: str) -> None:
    conv = convert_model(toy["path"], mode, toy["out"], x_calib=toy["x"])
    assert Path(conv["tflite_path"]).exists()
    assert conv["flash_bytes"] > 0

    ev = evaluate_tflite(conv["tflite_path"], toy["x"], toy["y"], max_samples=200)
    assert 0.0 <= ev["accuracy"] <= 1.0
    assert ev["num_samples"] == 200


def test_full_int8_reduces_ram(toy: dict) -> None:
    f32 = convert_model(toy["path"], "float32", toy["out"])
    i8 = convert_model(toy["path"], "full_int8", toy["out"], x_calib=toy["x"])
    assert i8["est_ram_bytes"] * 4 == f32["est_ram_bytes"]


def test_full_int8_requires_calibration(toy: dict) -> None:
    with pytest.raises(ValueError, match="calibration"):
        convert_model(toy["path"], "full_int8", toy["out"])


def test_check_constraints() -> None:
    target = {"flash_budget_kb": 10, "ram_budget_kb": 2}
    ok = check_constraints(5 * 1024, 1024, 0.955, target, reference_accuracy=0.96)
    assert ok["fits"] and ok["violations"] == []

    ko = check_constraints(20 * 1024, 4096, 0.94, target, reference_accuracy=0.96)
    assert not ko["fits"]
    assert len(ko["violations"]) == 3
