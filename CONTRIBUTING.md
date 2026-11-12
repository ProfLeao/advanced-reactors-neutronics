# Contributing

Each reactor should remain independently understandable and runnable. Place
its code, input files, tests, documentation, and reference figures in a
dedicated directory under `reactors/`.

Before submitting a model:

1. Document its sources, material definitions, geometry, boundary conditions,
   nuclear data requirements, and validation status.
2. Keep generated statepoints, restart files, and large numerical arrays out
   of Git. Include small, clearly identified figures or summaries when useful.
3. Add tests for geometry invariants and configuration validation. Mark tests
   that require external cross-section data.
4. Run the tests from the repository root and report any checks that could not
   be performed.
5. Confirm that the model's source, figures, and references may be
   redistributed under the repository's license.

Do not silently change a published reference case. Add a new named
configuration and document how its assumptions differ.
