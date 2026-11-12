# Model notes

## Geometry and materials

The core layout follows the six-ring arrangement reported by Aziz et al.
(2024): 90 fuel pins and 37 heat pipes, including the central pipe. Fuel is
uranium mononitride (UN); the input specifies an enrichment by ring. The model
also includes helium gaps, Mo-Re cladding and pipe walls, liquid sodium, and a
Nb-1Zr monolith.

The BeO reflector is represented in the geometry builder but is disabled in
`config/reference.yaml`. Changing `reflector.active` changes the physical
model and requires a separately justified criticality target; do not treat it
as a display-only option.

## Calculation

OpenMC runs in eigenvalue mode. The model tallies fission and heating on a
Cartesian mesh, total flux on the same mesh, thermal/fast flux separated at
0.625 eV, and a 40-group global spectrum. The extracted fission distribution
is normalized to 1 MW thermal for comparison and plotting. This normalization
does not determine the reactor's operating power.

Temperatures are interpolated over the configured range for Doppler feedback.
If a temperature field is supplied, the fuel temperature is averaged over
voxels with nonzero fission power; otherwise, the hottest 20% of the field is
used as a proxy. The heat-pipe and reflector temperatures are prescribed
values, not solved by this neutronics model.

## Limits

- The reference input has no active BeO reflector.
- The model does not calculate thermal hydraulics, heat-pipe performance,
  structural response, fuel depletion, or burnup.
- Cross-section data are an external input. Record the library and OpenMC
  versions when publishing results.
- Mesh tallies and pin-power reconstruction are discretized estimates.
- A Monte Carlo result is not, by itself, a safety assessment or design
  validation.

For comparison, Aziz et al. report a reference effective multiplication
factor and pin-power peaking values for their study. Differences in materials,
boundary conditions, nuclear data, and tally definitions must be assessed
before comparing numbers.
