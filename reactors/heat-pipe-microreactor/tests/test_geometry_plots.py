from pathlib import Path

import pytest

MODEL_ROOT = Path(__file__).resolve().parents[1]


def test_geometry_plotter_creates_slice_and_voxel_figures(tmp_path):
    pytest.importorskip("openmc")
    from heat_pipe_microreactor.geometry_plots import generate_geometry_plots

    output_dir = tmp_path / "geometry"
    outputs = generate_geometry_plots(
        MODEL_ROOT / "config" / "reference.yaml",
        output_dir,
        pixels=400,
        voxel_pixels=(20, 20, 10),
    )

    assert {path.name for path in outputs} == {
        "slice_core_xy.png",
        "slice_core_xz.png",
        "slice_core_yz.png",
        "slice_fuel_pin_zoom.png",
        "slice_heat_pipe_zoom.png",
        "voxel_plot_3d_render.png",
        "voxel_slices_orthogonal.png",
    }
    assert all(path.is_file() for path in outputs)
    assert not list(output_dir.glob("heat-pipe-geometry-*"))
