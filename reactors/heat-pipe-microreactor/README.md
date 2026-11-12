# Heat-Pipe Microreactor Neutronics

This directory contains an OpenMC eigenvalue model of a hexagonal,
heat-pipe-cooled microreactor. It is the neutronics portion of the
`microreactor_sim` study; it does not include that project's thermal solver or
multiphysics coupling.

## Model at a glance

- 90 uranium-mononitride (UN) fuel pins and 37 heat pipes
- Six fuel rings with enrichment values of 60, 62, 70, 70, 75, and 65 wt% U-235
- Sodium heat-pipe fluid, Mo-Re cladding and pipe walls, and a Nb-1Zr monolith
- Optional 10 cm BeO reflector; it is **off** in the reference configuration
- Eigenvalue calculation with mesh tallies for fission power and neutron flux,
  a two-group flux tally, and a 40-group energy spectrum

The model is based on the geometry and neutronics parameters in Aziz et al.
(2024). See [Model notes](docs/model-notes.md) for assumptions and limits.

## Install

From the repository root, create and activate the shared Conda environment,
then install this model in editable mode:

```bash
conda env create -f environment.yml
conda activate advanced-reactors-neutronics
python -m pip install -e reactors/heat-pipe-microreactor --no-deps
```

OpenMC cross-section data must be downloaded separately. One supported route
is:

```bash
python -m pip install openmc-data
python -m openmc_data.download.download_endf -d ~/openmc_data -r b8.0
```

Set `OPENMC_CROSS_SECTIONS` to the resulting `cross_sections.xml` file. In
PowerShell:

```powershell
$env:OPENMC_CROSS_SECTIONS = "C:\path\to\openmc_data\endf-b8.0-hdf5\cross_sections.xml"
```

In Bash:

```bash
export OPENMC_CROSS_SECTIONS="$HOME/openmc_data/endf-b8.0-hdf5/cross_sections.xml"
```

The same path can be set in `openmc.cross_sections_path` in a configuration
file. Do not commit the nuclear data library.

## Run

Run the reduced smoke case first:

```bash
heat-pipe-microreactor \
  --config reactors/heat-pipe-microreactor/config/smoke.yaml \
  --output-dir reactors/heat-pipe-microreactor/simulation-output/smoke
```

Run the reference case with:

```bash
heat-pipe-microreactor \
  --config reactors/heat-pipe-microreactor/config/reference.yaml \
  --output-dir reactors/heat-pipe-microreactor/simulation-output/reference
```

The run writes OpenMC statepoints and summary files, plus `power.npy` and
`flux.npy`, to the selected output directory. Existing statepoints are not
replaced unless `--overwrite` is given. Use `--resume` to continue from the
latest checkpoint. The optional `--seed` and `--temperature-K` arguments set
the random seed and a uniform initial material temperature.

Generate the standard flux, group-flux, spectrum, and pin-power figures from
an existing statepoint without rerunning transport:

```bash
heat-pipe-neutronics-plots \
  reactors/heat-pipe-microreactor/simulation-output/reference/statepoint.10000.h5 \
  --config reactors/heat-pipe-microreactor/config/reference.yaml \
  --output-dir reactors/heat-pipe-microreactor/simulation-output/plots
```

Generate 2D geometry slices and 3D voxel visualizations without running a
transport calculation:

```bash
heat-pipe-geometry-plots \
  --config reactors/heat-pipe-microreactor/config/reference.yaml \
  --output-dir reactors/heat-pipe-microreactor/simulation-output/geometry
```

The command writes five slice images and two voxel-rendering images. Its
temporary OpenMC XML and voxel data are deleted when the command finishes.
See [Geometry visualization](docs/geometry-visualization.md) for the latest
included figures and details.

## Tests

With the Conda environment active:

```bash
python -m pytest reactors/heat-pipe-microreactor/tests -q
```

The tests check the configuration, 90-pin/37-pipe layout, and data-path
handling. They do not run a transport calculation and do not validate nuclear
data.

## Reference results

The [reference-run summary](results/reference-run/README.md) contains selected
neutronics metrics and plots from the latest run available in the source
workspace. The raw statepoint and numerical arrays are not included. The
source analysis contains an inconsistent statepoint reference; see the
provenance note before using these figures as a reproducible benchmark.

## References

- Aziz, U. et al. (2024). “Non-uniform fuel distribution and thermo-mechanical
  analysis of a 1 MW thermal power micronuclear heat pipe reactor.” *Heliyon*,
  10(3), e25343. https://doi.org/10.1016/j.heliyon.2024.e25343
- Romano, P. K. et al. (2015). “OpenMC: A state-of-the-art Monte Carlo code
  for research and development.” *Annals of Nuclear Energy*, 82, 90–97.
  https://doi.org/10.1016/j.anucene.2014.07.048
