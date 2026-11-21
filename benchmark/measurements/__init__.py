"""Measurements behind protocol amendments that read the local training data.

They read `data/pairs/`, which is not in the repository, so they are not tests and CI
does not run them. Each prints its result. Run them from the repository root, for example
`uv run --frozen python -m benchmark.measurements.hiss_peak`. Python's `-I` flag is left
out on purpose: it removes the working directory from the import path, so `-m` could not
find the `benchmark` package. Run this way, the only code imported from the working
directory is this repository's own.
`engine_smoke` also imports the backend, so it runs with `PYTHONPATH=backend`.
"""
