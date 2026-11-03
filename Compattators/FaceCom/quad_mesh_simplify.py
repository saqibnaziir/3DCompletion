"""Real module (not a sys.modules alias) satisfying FaceCom's import.

FaceCom's utils/funcs.py does `from quad_mesh_simplify import simplify_mesh`.
In the 3dmesh env that same compiled extension is installed as `simplify`
(simplify.cp310-win_amd64.pyd), exporting an identical simplify_mesh, so the
decimation hierarchy built here is bit-for-bit what their code would build.

This has to be a file on disk rather than a sys.modules alias: the training
DataLoader uses num_workers > 0, and on Windows each worker is a fresh spawned
process that re-imports utils.my_dataset -> utils.funcs. An alias installed in
the parent process would not exist in those workers.
"""

from simplify import simplify_mesh  # noqa: F401

__all__ = ["simplify_mesh"]
