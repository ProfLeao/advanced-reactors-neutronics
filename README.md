# Advanced Reactors Neutronics

This repository collects standalone OpenMC neutronics models for advanced
reactors. Each reactor has its own directory under `reactors/`; shared
installation and contribution guidance lives at the repository root.

## Reactor models

| Model | Description |
| --- | --- |
| [Heat-pipe microreactor](reactors/heat-pipe-microreactor/README.md) | UN-fueled, hexagonal core with 90 fuel pins and 37 heat pipes. |

## Environment

The shared Conda environment installs OpenMC and the Python packages used by
the current model:

```bash
conda env create -f environment.yml
conda activate advanced-reactors-neutronics
python -m pip install -e reactors/heat-pipe-microreactor --no-deps
```

OpenMC cross-section data are not included. Follow the selected model's
instructions to download and configure them.

## Adding a reactor

Add a self-contained directory under `reactors/` with its own README,
configuration, source package, tests, and model-specific results. Add one row
to the table above. Move code to a shared root package only after at least two
models need the same stable behavior.

## Scope and license

These are research models, not validated tools for licensing, safety analysis,
or reactor design. Check each model's assumptions and validation notes before
using its results.

This repository is licensed under the BSD 3-Clause License. The license covers
only material whose contributors have the right to distribute; cited papers,
external software, and other third-party content retain their own terms.
Before publishing, confirm redistribution rights for the copied source and
figures, since the source workspace did not provide a license.
