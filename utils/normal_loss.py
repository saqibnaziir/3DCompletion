"""
Differentiable surface-normal loss for the completion diffusion model
(Olivier, 2026-07-30 thread; implemented 07-31).

Companion to utils/normals.py -- that module is the numpy METRIC used by
compare_arms.py, this one is the torch LOSS used during training. They compute
the same quantity (area-weighted vertex normals, angle to GT) so the number the
loss optimises and the number we report are the same thing.

WHY, from the 07-31 diagnostics (not a guess -- these were measured):
  * per-vertex position error and normal error are almost UNCORRELATED inside
    the hole (Spearman 0.08). The existing MSE loss is close to blind to
    orientation, so this term adds genuinely new signal.
  * our completions already have the RIGHT amount of high-frequency content
    (roughness ratio |L(ours)|/|L(gt)| = 0.996). So the fix is a MATCHING term,
    not a smoothness penalty -- a Laplacian/curvature penalty would push
    roughness below GT and make the surface worse while its own number improved.
  * ~73% of the orientation error is systematic (survives averaging 10 draws),
    which is the part this loss can actually reach. The other ~27% is sampling
    spread, i.e. the model's output diversity, which is a feature and is left
    alone deliberately.

USAGE NOTES, in order of how badly each bites if ignored:

1. UN-STANDARDIZE FIRST. The diffusion works on x0 divided by a PER-AXIS sigma.
   Anisotropic scaling does not preserve angles, so normals taken in the
   standardized space are not the surface's normals. `normal_cosine_loss` takes
   sigma and applies it internally -- do not pass pre-multiplied vertices.

2. WEIGHT BY alpha_bar_t. At high noise the predicted x0 is meaningless and its
   normals are worse than meaningless (differencing amplifies the noise). Pass
   `snr_weight=alphas_cumprod[t_node]`; it runs 1->0 with t and concentrates the
   term where x0_pred is reliable, with no extra hyperparameter.

3. 1 - cos, NEVER arccos. The derivative of arccos diverges at zero error --
   exactly where we want the model to converge. Degrees are for reporting.
"""

import torch

EPS = 1e-12


def batched_faces(faces, B, nodes_per_graph, device):
    """
    Replicate a single mesh's face list across a batch of B graphs.

    PyG concatenates graphs along the node axis, so graph b's vertices live at
    [b*N, (b+1)*N). The face list is identical for every sample (fixed topology),
    so it just needs the matching offset added.
    """
    f = faces.to(device=device, dtype=torch.long)
    offs = (torch.arange(B, device=device, dtype=torch.long) * nodes_per_graph)
    return (f.unsqueeze(0) + offs.view(B, 1, 1)).reshape(-1, 3)      # (B*F, 3)


def vertex_normals_torch(verts, faces_b):
    """
    Area-weighted unit vertex normals, (M,3), differentiable w.r.t. `verts`.

    The face cross products are left unnormalized on purpose: their magnitude is
    twice the triangle area, so scattering them onto the corners area-weights the
    average for free. Normalizing per face first would give a sliver the same
    vote as the large face beside it.
    """
    v0 = verts[faces_b[:, 0]]
    v1 = verts[faces_b[:, 1]]
    v2 = verts[faces_b[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)          # (Fb,3), area-weighted

    vn = torch.zeros_like(verts)
    for k in range(3):
        vn = vn.index_add(0, faces_b[:, k], fn)
    # clamp (not +EPS) so a degenerate vertex yields a finite, zero-ish normal
    # rather than a division that backprops NaN into the whole batch.
    return vn / vn.norm(dim=1, keepdim=True).clamp(min=1e-8)


def normal_cosine_loss(x0_pred_std, x0_gt_std, faces_b, sigma,
                       region_mask=None, snr_weight=None, vertex_weight=None):
    """
    mean over the selected region of (1 - <n_pred, n_gt>).

    Args:
        x0_pred_std: (M,3) predicted x0 in STANDARDIZED space (requires grad)
        x0_gt_std:   (M,3) ground-truth x0 in the same space
        faces_b:     (Fb,3) batched face index from `batched_faces`
        sigma:       (3,) per-axis std used to standardize; applied internally
        region_mask: (M,) bool, where to apply the loss (None = everywhere).
                     Normally the missing region -- the known region is
                     supervised by the data itself.
        snr_weight:  (M,) optional per-node weight, e.g. alphas_cumprod[t_node]
        vertex_weight: (M,) optional extra per-vertex weight (e.g. the geodesic
                     depth ramp), if you want the orientation term emphasised
                     the same way the positional term is

    Returns a scalar; exactly 0.0 (with grad) when the region is empty.
    """
    # 1. back to real units before taking any angle (see module docstring).
    p = x0_pred_std * sigma
    with torch.no_grad():                       # GT branch needs no gradient
        g = x0_gt_std * sigma
        n_gt = vertex_normals_torch(g, faces_b)
    n_pred = vertex_normals_torch(p, faces_b)

    per_vertex = 1.0 - (n_pred * n_gt).sum(dim=1)        # (M,), in [0,2]

    w = torch.ones_like(per_vertex)
    if snr_weight is not None:
        w = w * snr_weight
    if vertex_weight is not None:
        w = w * vertex_weight
    if region_mask is not None:
        w = w * region_mask.to(per_vertex.dtype)

    denom = w.sum()
    if float(denom) <= 0.0:
        return (per_vertex * 0.0).sum()          # keeps the graph connected
    return (w * per_vertex).sum() / denom


def normal_angle_degrees(x0_pred_std, x0_gt_std, faces_b, sigma, region_mask=None,
                         weight=None):
    """
    Mean angular error in DEGREES over the region -- for LOGGING only.

    Reported alongside the loss so the training curve is readable in the same
    units as compare_arms.py, without letting arccos anywhere near the gradient.

    `weight` (M,) MUST normally be the same alpha_bar_t the loss is weighted by.
    An UNWEIGHTED mean here is useless as a progress signal: it averages over all
    t, so it is dominated by high-noise steps where x0_pred is garbage and the
    angle is ~90 deg no matter how good the model is. Measured on the 07-31
    fine-tune it sat flat at ~49.5 deg for 16 epochs while the real test metric
    improved 5.5%. Pass the weight and it tracks the low-noise regime that the
    gradient actually cares about.
    """
    with torch.no_grad():
        p = x0_pred_std * sigma
        g = x0_gt_std * sigma
        cos = (vertex_normals_torch(p, faces_b) *
               vertex_normals_torch(g, faces_b)).sum(dim=1).clamp(-1.0, 1.0)
        ang = torch.rad2deg(torch.arccos(cos))

        w = torch.ones_like(ang) if weight is None else weight.to(ang.dtype).clone()
        if region_mask is not None:
            w = w * region_mask.to(ang.dtype)
        denom = w.sum()
        if float(denom) <= 0.0:
            return torch.zeros((), device=ang.device, dtype=ang.dtype)
        return (w * ang).sum() / denom
