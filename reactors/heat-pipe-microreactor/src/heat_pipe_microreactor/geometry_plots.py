"""Generate 2D slices and 3D voxel views of the OpenMC reactor geometry."""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import openmc
import yaml

from .geometry import (
    build_full_core,
    create_fuel_pin,
    create_heat_pipe,
    generate_hex_positions_aziz,
)
from .materials import (
    create_fuel_materials_by_ring,
    create_materials,
    nearest_supported_temperature,
)


_FUEL_COLOR = (214, 39, 40)
_MATERIAL_COLORS = {
    "helium": (255, 217, 47),
    "clad": (127, 127, 127),
    "sodium": (31, 119, 180),
    "hp_wall": (77, 77, 77),
    "monolith": (220, 220, 220),
    "reflector": (191, 227, 180),
}


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def _build_geometry(config: dict[str, Any]) -> tuple[
    openmc.Materials, openmc.Geometry, dict[str | int, openmc.Material]
]:
    openmc.reset_auto_ids()
    temperature = nearest_supported_temperature(
        config["openmc"]["temperature_default_K"]
    )
    materials = create_materials(
        temperature_K=temperature,
        enrichment_pct=config["fuel"]["enrichment"],
    )
    fuel_by_ring: dict[int, openmc.Material] = {}
    enrichment_by_ring = config["fuel"].get("enrichment_by_ring") or []
    if enrichment_by_ring:
        fuel_by_ring = create_fuel_materials_by_ring(
            enrichment_by_ring, temperature
        )
        materials.pop("fuel", None)
        materials.update(fuel_by_ring)

    positions = generate_hex_positions_aziz(
        pitch_cm=config["core"]["pitch_cm"]
    )
    fuel_universes: dict[int, openmc.Universe] = {}
    if fuel_by_ring:
        for ring, material in fuel_by_ring.items():
            fuel_universes[ring] = create_fuel_pin(
                radius_cm=config["fuel"]["radius_cm"],
                gap_outer_cm=config["clad"]["inner_radius_cm"],
                clad_outer_cm=config["clad"]["outer_radius_cm"],
                height_cm=config["core"]["active_height_cm"],
                fuel_material=material,
                gap_material=materials["helium"],
                clad_material=materials["clad"],
                monolith_material=materials["monolith"],
            )
        fuel_universe = next(iter(fuel_universes.values()))
    else:
        fuel_universe = create_fuel_pin(
            radius_cm=config["fuel"]["radius_cm"],
            gap_outer_cm=config["clad"]["inner_radius_cm"],
            clad_outer_cm=config["clad"]["outer_radius_cm"],
            height_cm=config["core"]["active_height_cm"],
            fuel_material=materials["fuel"],
            gap_material=materials["helium"],
            clad_material=materials["clad"],
            monolith_material=materials["monolith"],
        )

    heat_pipe_universe = create_heat_pipe(
        inner_radius_cm=config["heat_pipe"]["inner_radius_cm"],
        outer_radius_cm=config["heat_pipe"]["outer_radius_cm"],
        height_cm=config["core"]["active_height_cm"],
        fluid_material=materials["sodium"],
        wall_material=materials["hp_wall"],
    )
    geometry, _ = build_full_core(
        positions=positions,
        pitch_cm=config["core"]["pitch_cm"],
        fuel_universe=fuel_universe,
        fuel_universes=fuel_universes or None,
        hp_universe=heat_pipe_universe,
        monolith_material=materials["monolith"],
        reflector_material=materials["reflector"],
        height_cm=config["core"]["active_height_cm"],
        hex_apothem_cm=config["core"]["hex_side_cm"],
        reflector_thickness_cm=(
            config["reflector"]["thickness_cm"]
            if config["reflector"].get("active", False)
            else 0.0
        ),
    )
    return openmc.Materials(list(materials.values())), geometry, materials


def _ppm_to_png(ppm_path: Path, png_path: Path) -> None:
    with ppm_path.open("rb") as stream:
        if stream.readline().strip() != b"P6":
            raise ValueError(f"Unsupported PPM format in {ppm_path}")
        dimensions = stream.readline()
        while dimensions.startswith(b"#"):
            dimensions = stream.readline()
        width, height = map(int, dimensions.split())
        maximum = int(stream.readline().strip())
        pixels = np.frombuffer(stream.read(), dtype=np.uint8)
    expected_size = width * height * 3
    if pixels.size < expected_size:
        raise ValueError(f"Truncated PPM image: {ppm_path}")
    image = pixels[:expected_size].reshape(height, width, 3)
    if maximum != 255:
        image = np.rint(image.astype(float) * (255.0 / maximum)).astype(
            np.uint8
        )
    plt.imsave(png_path, image)


def _slice_plots(
    config: dict[str, Any],
    materials: dict[str | int, openmc.Material],
    fuel_materials: list[openmc.Material],
    output_pixels: int,
) -> list[openmc.SlicePlot]:
    core = config["core"]
    outer_apothem = core["hex_side_cm"]
    if config["reflector"].get("active", False):
        outer_apothem += config["reflector"]["thickness_cm"]
    half_width = outer_apothem * 1.3
    core_width = 2.0 * half_width
    height = core["active_height_cm"]
    colors = {material: _FUEL_COLOR for material in fuel_materials}
    colors.update({
        materials[name]: color for name, color in _MATERIAL_COLORS.items()
    })

    definitions = [
        ("core_xy", "xy", (0.0, 0.0, 0.0), (core_width, core_width),
         (output_pixels, output_pixels)),
        ("core_xz", "xz", (0.0, 0.0, 0.0), (core_width, height),
         (output_pixels, max(1, round(output_pixels * height / core_width)))),
        ("core_yz", "yz", (0.0, 0.0, 0.0), (core_width, height),
         (output_pixels, max(1, round(output_pixels * height / core_width)))),
        ("fuel_pin_zoom", "xy",
         (core["pitch_cm"], 0.0, 0.0), (4.0, 4.0),
         (output_pixels, output_pixels)),
        ("heat_pipe_zoom", "xy", (0.0, 0.0, 0.0), (3.5, 3.5),
         (output_pixels, output_pixels)),
    ]

    plots = []
    for plot_id, (name, basis, origin, width, pixels) in enumerate(
        definitions, start=1
    ):
        plot = openmc.SlicePlot(name=name)
        plot.id = plot_id
        plot.basis = basis
        plot.origin = origin
        plot.width = width
        plot.pixels = pixels
        plot.color_by = "material"
        plot.colors = colors
        plots.append(plot)
    return plots


def _plot_image(workdir: Path, plot: openmc.SlicePlot) -> Path:
    candidates = [
        workdir / f"plot_{plot.id}.png",
        workdir / f"{plot.name}.png",
        workdir / f"plot_{plot.id}.ppm",
        workdir / f"{plot.name}.ppm",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"OpenMC did not produce an image for slice {plot.name!r}"
    )


def _annotate_slice(
    image_path: Path,
    output_path: Path,
    plot: openmc.SlicePlot,
    labels: dict[str, tuple[int, int, int]],
) -> None:
    if image_path.suffix.lower() == ".ppm":
        converted = image_path.with_suffix(".png")
        _ppm_to_png(image_path, converted)
        image_path = converted

    if plot.basis == "xy":
        x_label, y_label = "X (cm)", "Y (cm)"
    elif plot.basis == "xz":
        x_label, y_label = "X (cm)", "Z (cm)"
    else:
        x_label, y_label = "Y (cm)", "Z (cm)"
    x_center, y_center = (
        (plot.origin[0], plot.origin[1])
        if plot.basis == "xy"
        else (plot.origin[0], plot.origin[2])
        if plot.basis == "xz"
        else (plot.origin[1], plot.origin[2])
    )
    extent = (
        x_center - plot.width[0] / 2,
        x_center + plot.width[0] / 2,
        y_center - plot.width[1] / 2,
        y_center + plot.width[1] / 2,
    )
    figure, axis = plt.subplots(figsize=(8, 7), dpi=160)
    axis.imshow(plt.imread(image_path), extent=extent, origin="upper")
    axis.set_title(f"OpenMC geometry: {plot.name.replace('_', ' ').title()}")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    handles = [
        mpatches.Patch(
            color=np.array(rgb) / 255.0,
            label=("UN fuel" if name.startswith("fuel") else name),
        )
        for name, rgb in labels.items()
    ]
    axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", dpi=160)
    plt.close(figure)


def _voxel_material_colors(
    material_ids: dict[str | int, int],
    fuel_ids: set[int],
) -> dict[int, tuple[float, float, float]]:
    colors = {
        -2: (1.0, 1.0, 1.0),
        material_ids["helium"]: (1.0, 0.85, 0.18),
        material_ids["clad"]: (0.5, 0.5, 0.5),
        material_ids["sodium"]: (0.12, 0.47, 0.71),
        material_ids["hp_wall"]: (0.3, 0.3, 0.3),
        material_ids["monolith"]: (0.85, 0.85, 0.85),
    }
    if "reflector" in material_ids:
        colors[material_ids["reflector"]] = (0.75, 0.89, 0.70)
    colors.update({material_id: (0.84, 0.15, 0.16) for material_id in fuel_ids})
    return colors


def _render_voxel_views(
    voxel_path: Path,
    output_dir: Path,
    material_ids: dict[str | int, int],
    fuel_ids: set[int],
) -> list[Path]:
    with h5py.File(voxel_path, "r") as voxel_file:
        data = voxel_file["data"][...]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D voxel data in {voxel_path}")

    nz, ny, nx = data.shape
    step_z = max(1, math.ceil(nz / 20))
    step_xy = max(1, math.ceil(nx / 35))
    voxels = data[::step_z, ::step_xy, ::step_xy].transpose(2, 1, 0)
    fuel_mask = np.isin(voxels, tuple(fuel_ids))
    sodium_mask = voxels == material_ids["sodium"]
    active_mask = fuel_mask | sodium_mask
    active_colors = np.zeros(active_mask.shape + (4,), dtype=float)
    active_colors[fuel_mask] = (0.84, 0.15, 0.16, 0.9)
    active_colors[sodium_mask] = (0.12, 0.47, 0.71, 0.9)

    figure = plt.figure(figsize=(14, 7), dpi=150)
    left = figure.add_subplot(121, projection="3d")
    left.voxels(active_mask, facecolors=active_colors, shade=False)
    left.set_title("Fuel pins and heat-pipe channels")
    left.set_xlabel("Voxel X")
    left.set_ylabel("Voxel Y")
    left.set_zlabel("Voxel Z")
    left.view_init(elev=28, azim=-55)

    right = figure.add_subplot(122, projection="3d")
    ix, iy, iz = np.indices(voxels.shape)
    cutaway = (ix <= voxels.shape[0] // 2) | (iy <= voxels.shape[1] // 2)
    monolith_mask = (
        (voxels == material_ids["monolith"])
        & cutaway
        & (
            (ix == 0)
            | (iy == 0)
            | (iz == 0)
            | (ix == voxels.shape[0] // 2)
            | (iy == voxels.shape[1] // 2)
        )
    )
    cutaway_mask = (fuel_mask | sodium_mask | monolith_mask) & cutaway
    cutaway_colors = np.zeros(cutaway_mask.shape + (4,), dtype=float)
    cutaway_colors[fuel_mask & cutaway] = (0.84, 0.15, 0.16, 0.95)
    cutaway_colors[sodium_mask & cutaway] = (0.12, 0.47, 0.71, 0.95)
    cutaway_colors[monolith_mask] = (0.75, 0.75, 0.75, 0.35)
    right.voxels(cutaway_mask, facecolors=cutaway_colors, shade=False)
    right.set_title("Monolith cutaway")
    right.set_xlabel("Voxel X")
    right.set_ylabel("Voxel Y")
    right.set_zlabel("Voxel Z")
    right.view_init(elev=28, azim=-55)

    output_3d = output_dir / "voxel_plot_3d_render.png"
    figure.tight_layout()
    figure.savefig(output_3d, bbox_inches="tight")
    plt.close(figure)

    color_map = _voxel_material_colors(material_ids, fuel_ids)

    def to_rgb(slice_data: np.ndarray) -> np.ndarray:
        rgb = np.ones(slice_data.shape + (3,), dtype=float)
        for material_id, color in color_map.items():
            rgb[slice_data == material_id] = color
        return rgb

    mid_z, mid_y, mid_x = nz // 2, ny // 2, nx // 2
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=160)
    axes[0].imshow(to_rgb(data[mid_z, :, :]), origin="lower")
    axes[0].set_title(f"XY slice at voxel z={mid_z}")
    axes[0].set_xlabel("Voxel X")
    axes[0].set_ylabel("Voxel Y")
    axes[1].imshow(to_rgb(data[:, mid_y, :]), origin="lower", aspect="auto")
    axes[1].set_title(f"XZ slice at voxel y={mid_y}")
    axes[1].set_xlabel("Voxel X")
    axes[1].set_ylabel("Voxel Z")
    axes[2].imshow(to_rgb(data[:, :, mid_x]), origin="lower", aspect="auto")
    axes[2].set_title(f"YZ slice at voxel x={mid_x}")
    axes[2].set_xlabel("Voxel Y")
    axes[2].set_ylabel("Voxel Z")
    figure.tight_layout()
    output_slices = output_dir / "voxel_slices_orthogonal.png"
    figure.savefig(output_slices, bbox_inches="tight")
    plt.close(figure)
    return [output_3d, output_slices]


def generate_geometry_plots(
    config_path: Path,
    output_dir: Path,
    pixels: int = 900,
    voxel_pixels: tuple[int, int, int] = (100, 100, 60),
) -> list[Path]:
    """Generate five material slices and two 3D voxel figures."""
    if pixels < 400:
        raise ValueError("Slice resolution must be at least 400 pixels")
    config = _load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    openmc_materials, geometry, material_map = _build_geometry(config)
    fuel_materials = [
        material
        for name, material in material_map.items()
        if name == "fuel" or isinstance(name, int)
    ]
    if not fuel_materials:
        raise ValueError("No fuel materials were created from the configuration")
    plots = _slice_plots(config, material_map, fuel_materials, pixels)
    voxel_plot = openmc.VoxelPlot(name="reactor_voxel")
    voxel_plot.id = 10
    voxel_plot.filename = "reactor_voxel"
    core = config["core"]
    outer_apothem = core["hex_side_cm"]
    if config["reflector"].get("active", False):
        outer_apothem += config["reflector"]["thickness_cm"]
    voxel_width = 2.0 * outer_apothem * 1.3
    voxel_plot.origin = (0.0, 0.0, 0.0)
    voxel_plot.width = (
        voxel_width, voxel_width, core["active_height_cm"]
    )
    voxel_plot.pixels = voxel_pixels
    voxel_plot.color_by = "material"
    plots.append(voxel_plot)

    labels = {"UN fuel": _FUEL_COLOR}
    labels.update(_MATERIAL_COLORS)
    material_ids = {name: material.id for name, material in material_map.items()}
    fuel_ids = {material.id for material in fuel_materials}

    with tempfile.TemporaryDirectory(
        prefix="heat-pipe-geometry-", dir=output_dir
    ) as temporary_directory:
        workdir = Path(temporary_directory)
        model = openmc.Model(geometry=geometry, materials=openmc_materials)
        model.export_to_xml(directory=str(workdir))
        plots.export_to_xml(str(workdir / "plots.xml"))
        openmc.plot_geometry(cwd=str(workdir), output=True)

        outputs = []
        for plot in plots[:-1]:
            image = _plot_image(workdir, plot)
            output_path = output_dir / f"slice_{plot.name}.png"
            _annotate_slice(image, output_path, plot, labels)
            outputs.append(output_path)

        voxel_path = workdir / "reactor_voxel.h5"
        if not voxel_path.is_file():
            voxel_path = workdir / "plot_10.h5"
        if not voxel_path.is_file():
            raise FileNotFoundError("OpenMC did not produce the voxel HDF5 file")
        outputs.extend(
            _render_voxel_views(
                voxel_path, output_dir, material_ids, fuel_ids
            )
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OpenMC geometry slice and voxel plots."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixels", type=int, default=900)
    args = parser.parse_args()
    outputs = generate_geometry_plots(args.config, args.output_dir, args.pixels)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
