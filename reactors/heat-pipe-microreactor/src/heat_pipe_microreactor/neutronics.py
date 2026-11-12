"""Run an OpenMC eigenvalue calculation and extract mesh-tally results."""

import argparse
import os
import time
import yaml
import numpy as np
import openmc
from pathlib import Path

from .materials import (
    create_materials,
    create_fuel_materials_by_ring,
    update_material_temperatures,
    nearest_supported_temperature,
)
from .geometry import (
    create_fuel_pin,
    create_heat_pipe,
    build_full_core,
    generate_hex_positions_aziz,
)


def resolve_cross_sections_path(cfg: dict) -> str:
    """Resolve the path to ``cross_sections.xml``.

    Check the configured path, ``OPENMC_CROSS_SECTIONS``, and a small set of
    conventional OpenMC data directories. Avoid recursively scanning a user's
    entire home directory.
    """
    candidates = []

    cfg_path = (cfg.get("openmc", {}) or {}).get("cross_sections_path", "")
    if cfg_path:
        candidates.append(Path(cfg_path))

    env_path = os.environ.get("OPENMC_CROSS_SECTIONS", "").strip()
    if env_path:
        candidates.append(Path(env_path))

    search_roots = [
        Path.cwd(),
        Path.home() / "openmc_data",
        Path.home() / ".local" / "share" / "openmc",
        Path.home() / ".config" / "openmc",
    ]
    for root in search_roots:
        candidates.extend((
            root / "cross_sections.xml",
            root / "endf-b8.0-hdf5" / "cross_sections.xml",
            root / "endfb-viii.0-hdf5" / "cross_sections.xml",
        ))

    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            os.environ["OPENMC_CROSS_SECTIONS"] = str(resolved)
            openmc.config["cross_sections"] = str(resolved)
            return str(resolved)

    raise FileNotFoundError(
        "No cross_sections.xml file was found. Set OPENMC_CROSS_SECTIONS or "
        "openmc.cross_sections_path to the library file. See the reactor "
        "README for download instructions."
    )


def ensure_cross_sections_available(cfg: dict) -> str:
    """Require a usable nuclear data library and explain how to configure it."""
    try:
        return resolve_cross_sections_path(cfg)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "OpenMC nuclear data were not found. Download the ENDF/B-VIII.0 "
            "library with openmc-data, then set OPENMC_CROSS_SECTIONS to its "
            "cross_sections.xml file."
        ) from exc


def load_config(config_path: str | Path | None = None) -> dict:
    """Load a YAML configuration, defaulting to this reactor's reference case."""
    if config_path is None:
        config_path = (Path(__file__).resolve().parents[2] /
                       "config" / "reference.yaml")
    config_file = Path(config_path).expanduser().resolve()
    with config_file.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_file}")
    return config


def setup_mesh_tally(cfg: dict) -> openmc.MeshFilter:
    """Create the 3D Cartesian mesh used for power and flux tallies.

    Parameters
    ----------
    cfg : dict
        Reactor configuration.

    Returns
    -------
    openmc.MeshFilter
        Mesh filter for use in tallies.
    """
    dims = cfg["openmc"]["mesh_tally_dim"]
    core_half = cfg["core"]["hex_side_cm"] * 2
    height = cfg["core"]["active_height_cm"]

    mesh = openmc.RegularMesh()
    mesh.dimension = dims
    mesh.lower_left = [-core_half, -core_half, -height / 2]
    mesh.upper_right = [core_half, core_half, height / 2]

    mesh_filter = openmc.MeshFilter(mesh)
    return mesh_filter


def create_tallies(mesh_filter: openmc.MeshFilter) -> list:
    """Create fission-power, flux-group, total-flux, and spectrum tallies.

    Parameters
    ----------
    mesh_filter : openmc.MeshFilter
        Cartesian mesh filter.

    Returns
    -------
    list of openmc.Tally
    """
    tallies = []

    # Fission and heating scores for the power tally.
    energy_filter = openmc.EnergyFilter([0.0, 20.0e6])  # [0, 20 MeV]

    fission_tally = openmc.Tally(name="power")
    fission_tally.filters = [mesh_filter, energy_filter]
    fission_tally.scores = ["fission", "heating"]
    tallies.append(fission_tally)

    # Total flux on the mesh.
    flux_tally = openmc.Tally(name="flux")
    flux_tally.filters = [mesh_filter]
    flux_tally.scores = ["flux"]
    tallies.append(flux_tally)

    # Two-group flux split at 0.625 eV.
    g2 = openmc.EnergyFilter([0.0, 0.625, 20.0e6])   # eV
    flux_g2 = openmc.Tally(name="flux_groups")
    flux_g2.filters = [mesh_filter, g2]
    flux_g2.scores = ["flux"]
    tallies.append(flux_g2)

    # Global 40-group spectrum, log-spaced from 1e-5 eV to 20 MeV.
    g_edges = np.geomspace(1.0e-5, 20.0e6, 41)
    spec = openmc.Tally(name="spectrum")
    spec.filters = [openmc.EnergyFilter(g_edges.tolist())]
    spec.scores = ["flux"]
    tallies.append(spec)

    return tallies


def _statepoint_batch(sp_path) -> int:
    """Return the batch number encoded in a ``statepoint.NNN.h5`` filename."""
    return int(Path(sp_path).stem.rsplit(".", 1)[-1])


def _newest_statepoint(output_path) -> Path:
    """Return the statepoint with the largest batch number in a directory."""
    sps = list(Path(output_path).glob("statepoint.*.h5"))
    if not sps:
        return None
    sps.sort(key=_statepoint_batch)
    return sps[-1]


def _extract_results(sp_path: str, cfg: dict,
                     elapsed_s: float = None) -> tuple:
    """Extract k-effective, power, and flux from a statepoint.

    This is used after a new run and when resuming from a completed run.
    """
    with openmc.StatePoint(sp_path) as sp:
        k_eff = sp.keff
        k_eff_sigma = float(k_eff.std_dev)
        print(f"\n  k_eff = {k_eff.nominal_value:.6f} ± {k_eff_sigma:.6f}")
        if elapsed_s is not None:
            print(f"  Run time: {elapsed_s:.1f} s")

        power_tally = sp.get_tally(name="power")
        if power_tally is None:
            raise RuntimeError("The statepoint does not contain the 'power' tally.")
        power_vals = power_tally.get_values(scores=["fission"]).flatten()

        flux_tally = sp.get_tally(name="flux")
        if flux_tally is None:
            raise RuntimeError("The statepoint does not contain the 'flux' tally.")
        flux_vals = flux_tally.get_values(scores=["flux"]).flatten()

        total_power = np.sum(power_vals)
        if total_power > 0:
            power_vals = power_vals / total_power * 1.0e6

    dims = cfg["openmc"]["mesh_tally_dim"]
    # OpenMC flattens mesh tallies in Fortran order, with x varying fastest.
    power_3d = power_vals.reshape(dims, order="F")
    flux_3d = flux_vals.reshape(dims, order="F")

    from .peaking import pin_power_peaking
    peaking = pin_power_peaking(power_3d, cfg)
    hottest = peaking["hottest"]
    print(f"  Pin peaking factor: {peaking['pin_peaking']:.3f} "
          f"(ring {hottest['ring']}, {hottest['rel_power']:.2f}x mean)")
    print(f"  Total power: {np.sum(power_3d) / 1e6:.3f} MWt")
    print(f"  Maximum local power: {np.max(power_3d):.1f} W/voxel")

    return (k_eff.nominal_value,
            k_eff_sigma,
            power_3d,
            flux_3d,
            str(sp_path))


def run_openmc(cfg: dict,
               temperature_field: np.ndarray = None,
               power_field: np.ndarray = None,
               temp_map: dict = None,
               output_dir: str | Path = "results/openmc",
               checkpoint_interval: int = 0,
               resume: bool = False,
               seed: int = None,
               overwrite: bool = False) -> tuple:
    """Run the full OpenMC neutronics calculation.

    Intermediate statepoints support checkpointing. Set ``resume=True`` to
    continue from the latest statepoint in ``output_dir``. A new run refuses
    to replace existing output unless ``overwrite=True`` is explicit.

    Parameters
    ----------
    cfg : dict
        Reactor configuration.
    temperature_field : np.ndarray, optional
        Temperature field used for Doppler feedback.
    power_field : np.ndarray, optional
        Previous power field, used to identify fissioning fuel voxels.
    output_dir : str | Path
        Directory for OpenMC output files.
    checkpoint_interval : int, optional
        Write an intermediate statepoint every N batches (0 disables it).
    resume : bool, optional
        Continue from an existing statepoint.
    seed : int, optional
        Random-number seed (random by default).
    overwrite : bool, optional
        Remove statepoints and summary files already in ``output_dir``.

    Returns
    -------
    tuple
        (k_eff, k_eff_sigma, power_distribution, flux_distribution,
         statepoint_path)
    """
    if resume and overwrite:
        raise ValueError("resume and overwrite cannot be used together.")

    print("=" * 60)
    print("  NEUTRONICS CALCULATION (OpenMC)")
    print("=" * 60)

    # Restart requires IDs to match those stored in the checkpoint.
    openmc.reset_auto_ids()

    cross_sections_path = ensure_cross_sections_available(cfg)
    print(f"  [OK] OpenMC data: {cross_sections_path}")

    # Create model materials.
    T_default = nearest_supported_temperature(cfg["openmc"]["temperature_default_K"])
    materials = create_materials(temperature_K=T_default,
                                 enrichment_pct=cfg["fuel"]["enrichment"])
    enrich_by_ring = (cfg.get("fuel", {}).get("enrichment_by_ring") or [])
    fuel_materials = None
    if enrich_by_ring:
        fuel_materials = create_fuel_materials_by_ring(enrich_by_ring, T_default)
        materials.pop("fuel", None)
        materials.update(fuel_materials)
        print(f"  [OK] Ring-wise fuel enrichment: {enrich_by_ring}%")

    if temp_map is not None:
        update_material_temperatures(materials, temp_map)
        if fuel_materials and "fuel" in temp_map:
            T_fuel = temp_map["fuel"]
            for ring in fuel_materials:
                fuel_materials[ring].temperature = T_fuel
        print("  [OK] Applied explicit Doppler temperature map.")
    elif temperature_field is not None:
        temp_map = _build_temperature_map(temperature_field, materials, cfg,
                                          power_field)
        update_material_temperatures(materials, temp_map)
        if fuel_materials:
            T_fuel = temp_map["fuel"]
            for ring in fuel_materials:
                fuel_materials[ring].temperature = T_fuel
        print(f"  [OK] Applied Doppler feedback. "
              f"T_fuel = {temp_map['fuel']:.1f} K, "
              f"T_avg = {np.mean(temperature_field):.1f} K")

    # Build the core geometry.
    print("  Building CSG core geometry...")

    # Build one fuel universe per ring when enrichment varies by ring.
    fuel_univ = None
    fuel_universes = {}
    if fuel_materials:
        for ring, mat in fuel_materials.items():
            fuel_universes[ring] = create_fuel_pin(
                radius_cm=cfg["fuel"]["radius_cm"],
                gap_outer_cm=cfg["clad"]["inner_radius_cm"],
                clad_outer_cm=cfg["clad"]["outer_radius_cm"],
                height_cm=cfg["core"]["active_height_cm"],
                fuel_material=mat,
                gap_material=materials["helium"],
                clad_material=materials["clad"],
                monolith_material=materials["monolith"],
            )
    else:
        fuel_univ = create_fuel_pin(
            radius_cm=cfg["fuel"]["radius_cm"],
            gap_outer_cm=cfg["clad"]["inner_radius_cm"],
            clad_outer_cm=cfg["clad"]["outer_radius_cm"],
            height_cm=cfg["core"]["active_height_cm"],
            fuel_material=materials["fuel"],
            gap_material=materials["helium"],
            clad_material=materials["clad"],
            monolith_material=materials["monolith"],
        )

    hp_univ = create_heat_pipe(
        inner_radius_cm=cfg["heat_pipe"]["inner_radius_cm"],
        outer_radius_cm=cfg["heat_pipe"]["outer_radius_cm"],
        height_cm=cfg["core"]["active_height_cm"],
        fluid_material=materials["sodium"],
        wall_material=materials["hp_wall"],
    )

    positions = generate_hex_positions_aziz(
        pitch_cm=cfg["core"]["pitch_cm"],
    )
    geometry, _ = build_full_core(
        positions=positions,
        pitch_cm=cfg["core"]["pitch_cm"],
        fuel_universe=fuel_univ,
        fuel_universes=fuel_universes or None,
        hp_universe=hp_univ,
        monolith_material=materials["monolith"],
        reflector_material=materials["reflector"],
        height_cm=cfg["core"]["active_height_cm"],
        hex_apothem_cm=cfg["core"]["hex_side_cm"],
        reflector_thickness_cm=(cfg["reflector"]["thickness_cm"]
                                if cfg["reflector"].get("active", False)
                                else 0.0),
    )
    print(f"  [OK] Placed {len(positions)} elements "
          f"(fuel pins and heat pipes).")

    # Set up the OpenMC run.
    n_batches = cfg["openmc"]["batches"]
    n_inactive = cfg["openmc"]["inactive"]
    if n_inactive >= n_batches:
        raise ValueError(
            f"Invalid OpenMC settings: batches={n_batches}, "
            f"inactive={n_inactive}; at least one active batch is required."
        )
    settings = openmc.Settings()
    settings.batches = n_batches
    settings.inactive = n_inactive
    settings.particles = cfg["openmc"]["particles"]
    settings.output = {"tallies": True}

    settings.run_mode = "eigenvalue"

    settings.temperature = {
        "default": T_default,
        "method": "interpolation",
        "range": tuple(cfg["openmc"].get(
            "temperature_range_K", (250.0, 2500.0))),
        "tolerance": 100.0,
    }

    if seed is None:
        seed = int(np.random.randint(1, 2**31))
    settings.seed = seed

    interval = int(checkpoint_interval or 0)
    if interval > 0:
        if interval >= n_batches:
            interval = max(1, n_batches // 2)
        settings.statepoint = {
            "batches": list(range(interval, n_batches + 1, interval))
        }

    # Start source particles in a box around the first fuel pin.
    source_box = openmc.stats.Box(
        [-cfg["core"]["pitch_cm"] / 2.0, -cfg["core"]["pitch_cm"] / 2.0, -cfg["core"]["active_height_cm"] / 2.0],
        [cfg["core"]["pitch_cm"] * 1.5, cfg["core"]["pitch_cm"] / 2.0, cfg["core"]["active_height_cm"] / 2.0],
    )
    settings.source = openmc.IndependentSource(space=source_box)
    settings.source_rejection_fraction = 0.05

    # Configure tallies.
    mesh_filter = setup_mesh_tally(cfg)
    tallies = create_tallies(mesh_filter)

    model = openmc.Model(
        geometry=geometry,
        materials=openmc.Materials(list(materials.values())),
        settings=settings,
        tallies=openmc.Tallies(tallies),
    )

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(exist_ok=True, parents=True)

    newest = _newest_statepoint(output_path) if resume else None
    resumed_batch = _statepoint_batch(newest) if newest is not None else 0
    if newest is not None and resumed_batch >= n_batches:
        print(f"\n  [resume] Run already completed at {newest.name} "
              f"(batch {resumed_batch}/{n_batches}); reusing its results.")
        return _extract_results(str(newest), cfg, elapsed_s=None)

    if resume and newest is None:
        print("\n  [resume] No statepoint found; starting a new run.")
        resume = False

    if not resume:
        existing_outputs = (
            list(output_path.glob("statepoint.*.h5")) +
            list(output_path.glob("summary.h5"))
        )
        if existing_outputs and not overwrite:
            raise FileExistsError(
                f"OpenMC output already exists in {output_path}. "
                "Use resume=True to continue or overwrite=True to replace it."
            )
        for old in existing_outputs:
            old.unlink()

    print(f"\n  Starting Monte Carlo: {n_batches} batches, "
          f"{cfg['openmc']['particles']} particles per batch")
    print(f"  Inactive batches: {n_inactive}")
    if interval > 0:
        print(f"  Checkpoint interval: {interval} batches")
    print("-" * 60)

    t_start = time.time()
    if resume and newest is not None:
        print(f"  [resume] Restarting from {newest.name} "
              f"(batch {resumed_batch}/{n_batches})...")
        sp_path = model.run(restart_file=str(newest), cwd=str(output_path))
    else:
        sp_path = model.run(cwd=str(output_path))
    t_elapsed = time.time() - t_start

    sp_final = _newest_statepoint(output_path)
    if sp_final is None and sp_path:
        sp_final = Path(sp_path)
    if sp_final is None or not Path(sp_final).exists():
        raise RuntimeError(
            "OpenMC did not produce a statepoint. Check the OpenMC output."
        )

    return _extract_results(str(sp_final), cfg, elapsed_s=t_elapsed)


def _build_temperature_map(temp_field: np.ndarray,
                            materials: dict,
                            cfg: dict,
                            power_field: np.ndarray = None) -> dict:
    """Map a temperature field to the material temperatures used by OpenMC.

    Fuel temperature is averaged over fissioning voxels when a power field is
    available; otherwise, the hottest 20% of the field is used.

    Parameters
    ----------
    temp_field : np.ndarray
        Three-dimensional temperature field.
    materials : dict
        Materials created for the model.
    cfg : dict
        Reactor configuration.
    power_field : np.ndarray, optional
        Previous power field, used to identify fissioning voxels.

    Returns
    -------
    dict
        Mapping from material role to temperature in kelvin.
    """
    T_avg = float(np.mean(temp_field))
    T_pipe = cfg["heat_pipe"]["operating_temp_K"]

    if power_field is not None and np.max(power_field) > 0:
        fiss_mask = power_field > 0
        if np.any(fiss_mask):
            T_fuel = float(np.mean(temp_field[fiss_mask]))
        else:
            T_fuel = T_avg
    else:
        flat = np.sort(np.asarray(temp_field).ravel())
        if flat.size > 0:
            T_fuel = float(np.mean(flat[-max(1, int(0.2 * flat.size)):]))
        else:
            T_fuel = T_avg

    t_min = cfg["openmc"].get("temperature_min_K", 250.0)
    t_max = cfg["openmc"].get("temperature_max_K", 2500.0)
    T_fuel = float(np.clip(T_fuel, t_min, t_max))
    T_mono = float(np.clip(T_avg, t_min, t_max))

    return {
        "fuel": T_fuel,
        "helium": T_fuel,
        "clad": T_fuel * 0.95,
        "sodium": T_pipe,
        "hp_wall": T_pipe * 0.9,
        "monolith": T_mono,
        "reflector": 600.0,
    }


def main():
    """Run the reactor model from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML input file")
    parser.add_argument("--output-dir", type=Path, default=Path("results/openmc"))
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--resume", action="store_true",
        help="Continue from the latest statepoint in the output directory"
    )
    output_group.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing statepoints and summary files"
    )
    parser.add_argument("--seed", type=int, help="Random-number seed")
    parser.add_argument("--temperature-K", type=float,
                        help="Uniform initial material temperature")
    args = parser.parse_args()

    cfg = load_config(args.config)
    temperature = (args.temperature_K if args.temperature_K is not None
                   else cfg["openmc"]["temperature_default_K"])
    temperature_field = np.full(
        tuple(cfg["openmc"]["mesh_tally_dim"]), temperature, dtype=float
    )
    result = run_openmc(
        cfg,
        temperature_field=temperature_field,
        output_dir=args.output_dir,
        checkpoint_interval=cfg["openmc"].get("checkpoint_interval", 0),
        resume=args.resume,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    output_path = args.output_dir.expanduser().resolve()
    np.save(output_path / "power.npy", result[2])
    np.save(output_path / "flux.npy", result[3])
    print(f"\nFinal result: k_eff = {result[0]:.6f} ± {result[1]:.6f}")
    print(f"Statepoint: {result[4]}")
    return result


if __name__ == "__main__":
    main()
