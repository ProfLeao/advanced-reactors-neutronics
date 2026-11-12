from pathlib import Path

import pytest
import yaml

MODEL_ROOT = Path(__file__).resolve().parents[1]


def test_reference_and_smoke_configs_describe_the_same_core():
    reference = yaml.safe_load(
        (MODEL_ROOT / "config" / "reference.yaml").read_text(encoding="utf-8")
    )
    smoke = yaml.safe_load(
        (MODEL_ROOT / "config" / "smoke.yaml").read_text(encoding="utf-8")
    )

    assert reference["core"] == smoke["core"]
    assert reference["fuel"] == smoke["fuel"]
    assert reference["reflector"] == smoke["reflector"]
    assert smoke["openmc"]["batches"] < reference["openmc"]["batches"]
    assert smoke["openmc"]["particles"] < reference["openmc"]["particles"]
    assert smoke["openmc"]["inactive"] < smoke["openmc"]["batches"]


def test_hex_layout_has_expected_fuel_and_heat_pipe_counts():
    pytest.importorskip("openmc")
    from heat_pipe_microreactor.geometry import generate_hex_positions_aziz

    positions = generate_hex_positions_aziz(3.03)
    fuel_count = sum(position[2] == "fuel" for position in positions)
    heat_pipe_count = sum(position[2] == "heatpipe" for position in positions)

    assert fuel_count == 90
    assert heat_pipe_count == 37


def test_load_config_uses_the_requested_path(tmp_path):
    pytest.importorskip("openmc")
    from heat_pipe_microreactor.neutronics import load_config

    config_path = tmp_path / "custom.yaml"
    config_path.write_text("openmc:\n  batches: 17\n", encoding="utf-8")

    assert load_config(config_path)["openmc"]["batches"] == 17


def test_cross_sections_path_can_be_set_with_environment(tmp_path, monkeypatch):
    pytest.importorskip("openmc")
    from heat_pipe_microreactor.neutronics import resolve_cross_sections_path

    cross_sections = tmp_path / "cross_sections.xml"
    cross_sections.write_text("<cross_sections />", encoding="utf-8")
    monkeypatch.setenv("OPENMC_CROSS_SECTIONS", str(cross_sections))

    assert resolve_cross_sections_path({}) == str(cross_sections.resolve())
