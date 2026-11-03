"""
Import shim for the FaceCom repo, so we never have to edit their source.

FaceCom's utils/funcs.py does `from quad_mesh_simplify import simplify_mesh`.
In the 3dmesh env that same compiled extension is installed under the name
`simplify` (simplify.cp310-win_amd64.pyd), exporting an identical
`simplify_mesh`. Aliasing it into sys.modules satisfies the import with the very
same binary, so the mesh-decimation hierarchy is bit-for-bit what their code
would build.

Import this BEFORE anything from the FaceCom repo:
    import _shim; _shim.install(FACECOM_ROOT)
"""

import sys
import types


def allow_full_unpickle():
    """Restore torch.load's pre-2.6 behaviour for the FaceCom/3DGeoMeshNet repos.

    PyTorch 2.6 changed the default of `weights_only` from False to True.
    FaceCom's utils/my_dataset.py calls `torch.load(path)` bare, and a PyG
    collate contains torch_geometric.data.data.DataEdgeAttr, which the
    restricted unpickler rejects. Their code cannot be fixed from outside, so we
    restore the old default for the process.

    This is safe here and only here: every file loaded through this path is one
    we generated locally (processed/*.pt from stage_coma_trainset.py) or a
    checkpoint already on disk. Do not reuse this for downloaded weights.

    Preferred narrower fix, applied first when the classes are importable:
    allowlist just the PyG attribute classes.
    """
    import torch

    try:
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage
        torch.serialization.add_safe_globals(
            [DataEdgeAttr, DataTensorAttr, GlobalStorage])
    except Exception:
        pass

    if getattr(torch.load, "_full_unpickle_patched", False):
        return
    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _load._full_unpickle_patched = True
    torch.load = _load


def install(facecom_root=None):
    if "quad_mesh_simplify" not in sys.modules:
        try:
            import simplify
        except ImportError as e:
            raise ImportError(
                "Neither 'quad_mesh_simplify' nor 'simplify' is importable. "
                "The FaceCom/3DGeoMeshNet mesh hierarchy cannot be built."
            ) from e
        shim = types.ModuleType("quad_mesh_simplify")
        shim.simplify_mesh = simplify.simplify_mesh
        sys.modules["quad_mesh_simplify"] = shim

    if facecom_root and facecom_root not in sys.path:
        # Must precede 3Dis on the path: 3Dis also ships utils/ and config/
        # packages, and the wrong one silently loads a different architecture.
        sys.path.insert(0, facecom_root)
