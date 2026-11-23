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


def test_grounding_accepts_verbatim_quote_with_sizes(chip_scenario: scenario_mod.Scenario) -> None:
    chip_scenario.retrieved = ["| Memory configuration\n | 8 = 64 Kbyte flash/16 Kbyte RAM\n | B = 128 Kbyte"]
    assert chip_scenario.check_grounding("8 = 64 Kbyte flash/16 Kbyte RAM", 64, 16, True) == []
    # MB values count in KB
    chip_scenario.retrieved.append("G = 1 MB of flash memory, 128 KB SRAM")
    assert chip_scenario.check_grounding("G = 1 MB of flash memory ... 128 KB SRAM", 1024, 128, True) == []


def test_grounding_rejects_invented_quote_or_wrong_numbers(chip_scenario: scenario_mod.Scenario) -> None:
    chip_scenario.retrieved = ["8 = 64 Kbyte flash/16 Kbyte RAM"]
    problems = chip_scenario.check_grounding("8 = 256 Kbyte flash/32 Kbyte RAM", 256, 32, True)
    assert any("not found verbatim" in p for p in problems)
    problems = chip_scenario.check_grounding("8 = 64 Kbyte flash/16 Kbyte RAM", 256, 32, True)
    assert len(problems) == 2


def test_rejected_declaration_leaves_budget_unset(chip_scenario: scenario_mod.Scenario) -> None:
    out = chip_scenario.set_memory_budget(0, 512, False, "p.1", evidence_quote="made up 512 KB")
    assert "error" in out
    assert chip_scenario.target is None
    assert chip_scenario.rejected_declarations == 1
