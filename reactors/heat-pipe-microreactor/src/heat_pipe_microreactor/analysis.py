"""Extract OpenMC statepoint tallies and create neutronics plots."""

import argparse
from pathlib import Path

import numpy as np
import openmc

from .neutronics import load_config
from .plot_neutronics import (
    plot_flux_group_maps,
    plot_flux_slices,
    plot_pin_power_map,
    plot_spectrum,
)


def _required_tally(statepoint, name: str):
    try:
        tally = statepoint.get_tally(name=name)
    except LookupError as exc:
        raise RuntimeError(
            f"Statepoint is missing the required '{name}' tally."
        ) from exc
    if tally is None:
        raise RuntimeError(f"Statepoint is missing the required '{name}' tally.")
    return tally


def load_tally_fields(statepoint_path: str | Path, cfg: dict) -> dict:
    """Load mean and one-sigma fields from a completed statepoint."""
    path = Path(statepoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Statepoint does not exist: {path}")

    dimensions = tuple(cfg["openmc"]["mesh_tally_dim"])
    fields = {"statepoint": str(path)}

    def relative_uncertainty(mean, sigma):
        return np.divide(
            sigma,
            mean,
            out=np.zeros_like(sigma, dtype=float),
            where=mean > 0,
        )

    with openmc.StatePoint(str(path)) as statepoint:
        fields["k_eff"] = float(statepoint.keff.nominal_value)
        fields["k_eff_sigma"] = float(statepoint.keff.std_dev)

        power_tally = _required_tally(statepoint, "power")
        power_mean = np.asarray(
            power_tally.get_values(scores=["fission"])
        ).ravel()
        power_sigma = np.asarray(
            power_tally.get_values(scores=["fission"], value="std_dev")
        ).ravel()
        fields["power_mean"] = power_mean.reshape(dimensions, order="F")
        fields["power_sigma"] = power_sigma.reshape(dimensions, order="F")

        flux_tally = _required_tally(statepoint, "flux")
        flux_mean = np.asarray(
            flux_tally.get_values(scores=["flux"])
        ).ravel()
        flux_sigma = np.asarray(
            flux_tally.get_values(scores=["flux"], value="std_dev")
        ).ravel()
        fields["flux_mean"] = flux_mean.reshape(dimensions, order="F")
        fields["flux_sigma"] = flux_sigma.reshape(dimensions, order="F")
        fields["flux_rel"] = relative_uncertainty(
            flux_mean, flux_sigma
        ).reshape(dimensions, order="F")

        groups = _required_tally(statepoint, "flux_groups")
        group_mean = np.squeeze(np.asarray(
            groups.get_values(scores=["flux"])
        )).reshape(-1, 2)
        group_sigma = np.squeeze(np.asarray(
            groups.get_values(scores=["flux"], value="std_dev")
        )).reshape(-1, 2)
        fields["flux_thermal"] = group_mean[:, 0].reshape(
            dimensions, order="F"
        )
        fields["flux_fast"] = group_mean[:, 1].reshape(
            dimensions, order="F"
        )
        fields["flux_fast_rel"] = relative_uncertainty(
            group_mean[:, 1], group_sigma[:, 1]
        ).reshape(dimensions, order="F")

        spectrum_tally = _required_tally(statepoint, "spectrum")
        fields["spectrum_mean"] = np.asarray(
            spectrum_tally.get_values(scores=["flux"])
        ).squeeze().astype(float)
        fields["spectrum_sigma"] = np.asarray(
            spectrum_tally.get_values(scores=["flux"], value="std_dev")
        ).squeeze().astype(float)
        energy_filter = next(
            filter_ for filter_ in spectrum_tally.filters
            if isinstance(filter_, openmc.EnergyFilter)
        )
        fields["energy_edges_eV"] = np.asarray(energy_filter.bins, dtype=float)

    return fields


def plot_statepoint(statepoint_path: str | Path, cfg: dict,
                    output_dir: str | Path) -> list[str]:
    """Generate the four standard neutronics figures from a statepoint."""
    fields = load_tally_fields(statepoint_path, cfg)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    return [
        plot_flux_slices(
            fields["flux_mean"], cfg, str(output),
            flux_rel=fields["flux_rel"],
        ),
        plot_flux_group_maps(
            fields["flux_thermal"], fields["flux_fast"], cfg, str(output),
            rel_fast=fields["flux_fast_rel"],
        ),
        plot_spectrum(
            fields["energy_edges_eV"], fields["spectrum_mean"], cfg,
            str(output), phi_std=fields["spectrum_sigma"],
        ),
        plot_pin_power_map(fields["power_mean"], cfg, str(output)),
    ]


def main():
    """Create plots from an existing OpenMC statepoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("statepoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("plots-neutronics"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = plot_statepoint(args.statepoint, cfg, args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
