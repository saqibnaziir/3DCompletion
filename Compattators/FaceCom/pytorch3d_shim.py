"""
Minimal stand-ins for the pytorch3d entry points FaceCom's code imports.

pytorch3d has no Windows wheels and needs CUDA toolkit + MSVC to build, so it is
absent from the 3dmesh env. Rather than rewrite FaceCom's completion pipeline
around that, we register these modules in sys.modules and then import their
utils/completion.py unmodified -- their rigid registration, their fit loop,
their post-processing all run as written.

Only what FaceCom actually touches is implemented:

  point_face_distance      exact squared point-to-triangle distance, the one
                           piece that carries a gradient into the generator.
                           Computed in two passes: an argmin search under
                           no_grad (chunked, so memory stays flat), then the
                           differentiable distance to the winning triangle
                           only. min() has the gradient of its argmin branch, so
                           this is exact, not an approximation.
  mesh_laplacian_smoothing uniform-weight Laplacian magnitude. FaceCom computes
                           it inside projection() but discards it (`loss =
                           loss_dfrm`), so only its callability matters.
  Meshes / Pointclouds     just enough of the container API for
                           utils/pytorch3d_extend.py.
  renderer, chamfer        never reached on the completion path; present so the
                           imports at the top of completion.py resolve.

Install BEFORE importing anything from FaceCom_repo:
    import pytorch3d_shim; pytorch3d_shim.install()
"""

import sys
import types

import torch

CHUNK = 512          # points per no-grad chunk in the argmin search


def _closest_point_on_triangle(p, a, b, c):
    """Ericson, Real-Time Collision Detection sec. 5.1.5. Broadcasting."""
    ab, ac = b - a, c - a
    ap = p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    eps = 1e-20

    denom = 1.0 / (va + vb + vc).clamp_min(eps)
    out = a + ab * (vb * denom).unsqueeze(-1) + ac * (vc * denom).unsqueeze(-1)

    def pick(cond, pt):
        return torch.where(cond.unsqueeze(-1), pt, out)

    out = pick((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
               b + (c - b) * ((d4 - d3)
                              / (d4 - d3 + d5 - d6).clamp_min(eps)).unsqueeze(-1))
    out = pick((vb <= 0) & (d2 >= 0) & (d6 <= 0),
               a + ac * (d2 / (d2 - d6).clamp_min(eps)).unsqueeze(-1))
    out = pick((vc <= 0) & (d1 >= 0) & (d3 <= 0),
               a + ab * (d1 / (d1 - d3).clamp_min(eps)).unsqueeze(-1))
    out = pick((d6 >= 0) & (d5 <= d6), c)
    out = pick((d3 >= 0) & (d4 <= d3), b)
    out = pick((d1 <= 0) & (d2 <= 0), a)
    return out


def point_face_distance(points, points_first_idx, tris, tris_first_idx,
                        max_points, min_triangle_area=1e-6):
    """(P,) squared distance from each point to the nearest triangle.

    Signature matches pytorch3d's. Single-mesh single-cloud only, which is all
    FaceCom's pytorch3d_extend.py ever constructs.
    """
    with torch.no_grad():
        best = None
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        for s in range(0, points.shape[0], CHUNK):
            p = points[s:s + CHUNK].unsqueeze(1)                  # (n,1,3)
            q = _closest_point_on_triangle(p, a, b, c)            # (n,T,3)
            d = ((p - q) ** 2).sum(-1)                            # (n,T)
            idx = d.argmin(dim=1)
            best = idx if best is None else torch.cat([best, idx])

    win = tris[best]                                              # (P,3,3)
    q = _closest_point_on_triangle(points, win[:, 0], win[:, 1], win[:, 2])
    # FaceCom takes sqrt() of this, whose gradient is infinite at 0. Our closest
    # point is returned exactly (`out = a` for the vertex regions), so a point
    # sitting on the surface gives a hard 0 and NaNs their optimizer; pytorch3d's
    # float32 kernel lands on ~1e-14 there instead. Clamp to that same floor.
    return ((points - q) ** 2).sum(-1).clamp_min(1e-14)


def mesh_laplacian_smoothing(meshes, method="uniform"):
    v = meshes.verts_packed()
    f = meshes.faces_packed().long()
    n = v.shape[0]
    i = torch.cat([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = torch.cat([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    acc = torch.zeros_like(v).index_add_(0, i, v[j])
    deg = torch.zeros(n, device=v.device, dtype=v.dtype).index_add_(
        0, i, torch.ones_like(i, dtype=v.dtype))
    return (acc / deg.clamp_min(1).unsqueeze(-1) - v).norm(dim=1).mean()


def chamfer_distance(x, y, **kw):
    d = torch.cdist(x, y)
    return d.min(-1).values.mean() + d.min(-2).values.mean(), None


class Meshes:
    def __init__(self, verts, faces):
        self._v, self._f = list(verts), list(faces)

    def __len__(self):
        return len(self._v)

    def verts_packed(self):
        return self._v[0]

    def faces_packed(self):
        return self._f[0]

    def mesh_to_faces_packed_first_idx(self):
        return torch.zeros(1, dtype=torch.long, device=self._f[0].device)

    def num_faces_per_mesh(self):
        return torch.tensor([self._f[0].shape[0]], device=self._f[0].device)


class Pointclouds:
    def __init__(self, points):
        self._p = list(points)

    def __len__(self):
        return len(self._p)

    def points_packed(self):
        return self._p[0]

    def cloud_to_packed_first_idx(self):
        return torch.zeros(1, dtype=torch.long, device=self._p[0].device)

    def num_points_per_cloud(self):
        return torch.tensor([self._p[0].shape[0]], device=self._p[0].device)


def install():
    if "pytorch3d" in sys.modules:
        return

    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    root = mod("pytorch3d")
    root.structures = mod("pytorch3d.structures", Meshes=Meshes,
                          Pointclouds=Pointclouds)
    pmd = mod("pytorch3d.loss.point_mesh_distance",
              point_face_distance=point_face_distance)
    root.loss = mod("pytorch3d.loss",
                    mesh_laplacian_smoothing=mesh_laplacian_smoothing,
                    chamfer_distance=chamfer_distance,
                    point_mesh_distance=pmd)
    # never called on the completion path; only imported
    root.renderer = mod("pytorch3d.renderer",
                        **{n: object for n in (
                            "look_at_view_transform", "FoVPerspectiveCameras",
                            "PointLights", "RasterizationSettings",
                            "MeshRenderer", "MeshRasterizer", "SoftPhongShader",
                            "TexturesVertex")})
