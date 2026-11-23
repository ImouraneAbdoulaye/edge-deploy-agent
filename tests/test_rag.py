"""Chunking and budget policy tests (no PDF, no embedding model)."""

import pytest

from edge_deploy_agent.agent import scenario as scenario_mod
from edge_deploy_agent.rag.index import chunk_page, load_chips


def test_chunk_page_overlaps_and_covers_all_lines() -> None:
    page = "\n".join(f"line {i} " + "x" * 90 for i in range(40))
    chunks = chunk_page(page, max_chars=1000, overlap_lines=3)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 + 110 for c in chunks)
    # consecutive chunks share their boundary lines
    assert chunks[0].splitlines()[-3:] == chunks[1].splitlines()[:3]
    # no line lost
    assert {ln for c in chunks for ln in c.splitlines()} == set(page.splitlines())


def test_every_chip_has_ground_truth() -> None:
    for chip, cfg in load_chips().items():
        gt = cfg["ground_truth"]
        assert gt["sram_kb"] > 0, chip
        assert (gt["flash_kb"] > 0) == gt["internal_flash"], chip


@pytest.fixture
def chip_scenario(monkeypatch: pytest.MonkeyPatch) -> scenario_mod.Scenario:
    monkeypatch.setattr(scenario_mod, "reference_accuracy", lambda model, backend: 0.9)
    return scenario_mod.Scenario("mnist_cnn_small", "MIMXRT1062")


def test_deploy_refused_before_budget(chip_scenario: scenario_mod.Scenario) -> None:
    assert chip_scenario.target is None
    assert "set_memory_budget" in chip_scenario.deploy_and_check("qualia", "int8")["error"]


def test_budget_policy(chip_scenario: scenario_mod.Scenario) -> None:
    out = chip_scenario.set_memory_budget(flash_kb=0, sram_kb=1024, internal_flash=False, evidence="p.1")
    assert out == {
        "model_flash_budget_kb": scenario_mod.EXTERNAL_FLASH_BUDGET_KB,
        "model_ram_budget_kb": 1024 * scenario_mod.MODEL_RAM_SHARE,
    }
    out = chip_scenario.set_memory_budget(flash_kb=630, sram_kb=320, internal_flash=True, evidence="p.2")
    assert out["model_flash_budget_kb"] == 630 * scenario_mod.MODEL_FLASH_SHARE
