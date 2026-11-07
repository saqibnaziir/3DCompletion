"""
Figures for the "Diversity and calibrated per-vertex uncertainty" subsection.

Emits four standalone figures from the K completions dumped by
research/dump_hero_samples.py:

  fig1_samples        3D panels only: input | 3 sample deviations | mean-of-K
                      | predicted uncertainty u | actual error e
  fig2_scatter        per-vertex u against e
  fig3_sparsification sparsification curves against the oracle
  fig4_reliability    predicted vs actual magnitude per uncertainty decile

  python research/fig_uncertainty_hero.py --npz hero.npz --outdir figs/
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import spearmanr

# --------------------------------------------------------------------------- #
# styling
# --------------------------------------------------------------------------- #
INK   = '#14161a'
MUTED = '#6b7280'
GRID  = '#dfe3e8'
SKIN  = np.array([0.855, 0.827, 0.800])
HOLE  = np.array([0.40, 0.44, 0.50])
ACC   = ['#C2410C', '#1D6F8B', '#4B7F52', '#7A4E9E']

# u and e are magnitudes on the same mm scale as the error panels in the
# qualitative / ladder / DFAUST figures, so they use the SAME colormap (jet).
# 0 is the bottom of the scale (dark blue), which is what makes "u is exactly
# zero where we observed" visible rather than asserted -- the observed region
# reads dark exactly as it does in every other error panel in the paper.
CMAP_UE = plt.get_cmap('jet')
# sample - mean is SIGNED (inward/outward along the normal). It needs a
# diverging map with a neutral zero; jet has no meaningful midpoint, so this
# one stays diverging on purpose -- a different quantity, not an inconsistency.
CMAP_DEV = LinearSegmentedColormap.from_list(
    'dev', ['#1D6F8B', '#7FB3C4', '#EFECE6', '#E7A08A', '#C2410C'])

LABEL_BOX = dict(boxstyle='round,pad=0.28', fc='white', ec='none', alpha=0.88)
BADGE_BOX = dict(boxstyle='round,pad=0.30', fc='white', ec='#d7dbe0', lw=0.6,
                 alpha=0.95)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'text.color': INK,
    'axes.edgecolor': '#c8cdd3', 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.labelsize': 8.5, 'axes.titlesize': 9, 'figure.dpi': 200,
    'savefig.facecolor': 'white',
})


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def rotate(V, yaw=0.0, pitch=0.0):
    a, b = np.deg2rad(yaw), np.deg2rad(pitch)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return V @ Ry.T @ Rx.T


def vertex_normals(V, F):
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn = fn / np.clip(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12, None)
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    return vn / np.clip(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12, None)


def _shade(V, F):
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn = fn / np.clip(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12, None)
    vn = vertex_normals(V, F)
    l = np.array([0.32, 0.42, 1.0]); l /= np.linalg.norm(l)
    lam = (0.44 + 0.56 * np.clip(vn @ l, 0, 1) ** 0.9
           + 0.16 * np.clip(vn @ l, 0, 1) ** 22)[F].mean(axis=1)
    return np.clip(lam, 0, 1.28), fn[:, 2] > 0


def draw(ax, V, F, rgb_v, yaw=-35.0, pitch=0.0, outline=None, headroom=0.0,
         outline_color='#2b3138'):
    """Painter's algorithm, back-faces culled. `outline` traces a vertex mask's
    boundary; `headroom` adds blank space on top for labels."""
    Vr = rotate(V, yaw, pitch)
    lam, front = _shade(Vr, F)
    tri = Vr[F]
    rgb = np.clip(rgb_v[F].mean(axis=1) * lam[:, None], 0, 1)

    order = np.argsort(tri[front, :, 2].mean(axis=1))
    ax.add_collection(PolyCollection(tri[front][order][:, :, :2],
                                     facecolors=rgb[front][order],
                                     edgecolors=rgb[front][order],
                                     linewidths=0.06, antialiaseds=True))
    if outline is not None:
        Ff, segs = F[front], []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            va, vb = Ff[:, a], Ff[:, b]
            sel = outline[va] != outline[vb]           # edge straddles the rim
            if sel.any():
                segs.append(np.stack([Vr[va[sel], :2], Vr[vb[sel], :2]], axis=1))
        if segs:
            ax.add_collection(LineCollection(np.concatenate(segs),
                                             colors=outline_color,
                                             linewidths=0.38, alpha=0.7, zorder=5))

    x0, x1 = Vr[:, 0].min(), Vr[:, 0].max()
    y0, y1 = Vr[:, 1].min(), Vr[:, 1].max()
    padx, pady = 0.03 * (x1 - x0), 0.02 * (y1 - y0)
    ax.set_xlim(x0 - padx, x1 + padx)
    ax.set_ylim(y0 - pady, y1 + pady + headroom * (y1 - y0))
    ax.set_aspect('equal'); ax.axis('off')


def scalar_rgb(s, cmap, vmin, vmax):
    """Colour the WHOLE mesh by its true value, observed region included."""
    return cmap(np.clip((s - vmin) / max(vmax - vmin, 1e-12), 0, 1))[:, :3]


def sparsification(u, e, n=100):
    base = max(e.mean(), 1e-12)
    fr = np.linspace(0, 1, n, endpoint=False)
    eu, eo = e[np.argsort(-u)], e[np.argsort(-e)]
    cut = (fr * len(e)).astype(int)
    s = np.array([eu[k:].mean() / base if k < len(e) else 0 for k in cut])
    o = np.array([eo[k:].mean() / base if k < len(e) else 0 for k in cut])
    return fr, s, o


def load(npz):
    d = np.load(npz)
    F = d['faces']
    names = [str(m) for m in d['meshes']]
    K = int(d['K'])
    out = []
    for i, name in enumerate(names):
        stack = d[f'{name}_stack']
        gt, mask = d[f'{name}_gt'], d[f'{name}_mask'].astype(bool)
        mu = stack.mean(axis=0)
        u = np.linalg.norm(stack.std(axis=0), axis=1) * 1000.0
        e = np.linalg.norm(mu - gt, axis=1) * 1000.0
        out.append(dict(name=name, label=name.replace('test_', 'mesh '),
                        colour=ACC[i % len(ACC)], stack=stack, gt=gt, mask=mask,
                        mu=mu, u=u, e=e, um=u[mask], em=e[mask],
                        miss=100 * mask.sum() / len(mask),
                        rho=spearmanr(u[mask], e[mask])[0]))
    return F, K, out


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def save(fig, outdir, stem):
    for ext in ('png', 'pdf'):
        p = os.path.join(outdir, f'{stem}.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {stem}.png / .pdf')


# --------------------------------------------------------------------------- #
# figure 1 - 3D panels only
# --------------------------------------------------------------------------- #
def figure_samples(F, K, m, outdir, yaw, sample_ids):
    """Compact layout for a single mesh:

        +---------+  Sample 1 | Sample 2
        |  Input  |  Sample 3 | Mean of K
        +---------+     u     |     e
    """
    mask, mu, u, e = m['mask'], m['mu'], m['u'], m['e']
    nrm = vertex_normals(mu, F)
    vmax = float(np.percentile(np.concatenate([m['um'], m['em']]), 97))
    dev_all = ((m['stack'] - mu) * 1000.0 * nrm).sum(-1)
    dmax = float(np.percentile(np.abs(dev_all[:, mask]), 98))

    fig = plt.figure(figsize=(5.5, 5.2))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.52, 1, 1, 0.058],
                          wspace=0.02, hspace=0.24,
                          left=0.008, right=0.935, top=0.935, bottom=0.012)

    def tag(ax, text, sub=None):
        """Title in points above the axes, so short panels do not collide."""
        ax.annotate(text, xy=(0.5, 1.0), xycoords='axes fraction',
                    xytext=(0, 12 if sub else 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8.4, color=INK,
                    fontweight='medium')
        if sub:
            ax.annotate(sub, xy=(0.5, 1.0), xycoords='axes fraction',
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=6.9, color=MUTED)

    # ---- input, spanning the full height ---------------------------------- #
    ax = fig.add_subplot(gs[:, 0])
    col = np.tile(SKIN, (len(mu), 1)); col[mask] = HOLE
    draw(ax, m['gt'], F, col, yaw=yaw)
    tag(ax, 'Input', f"{m['miss']:.1f}% of vertices missing")

    # ---- six result panels ------------------------------------------------- #
    cells = [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)]
    titles = ['Sample 1 $-$ mean', 'Sample 2 $-$ mean', 'Sample 3 $-$ mean',
              'Mean of $K$', 'Predicted uncertainty $u$', 'Actual error $e$']
    subs = [None, None, None, None, 'no ground truth', 'needs ground truth']

    axs = []
    for (r, c), t, s in zip(cells, titles, subs):
        a = fig.add_subplot(gs[r, c]); tag(a, t, s); axs.append(a)

    for k, si in enumerate(sample_ids):                 # samples 1, 2, 3
        draw(axs[k], m['stack'][si], F,
             scalar_rgb(dev_all[si], CMAP_DEV, -dmax, dmax),
             yaw=yaw, outline=mask)

    draw(axs[3], mu, F, np.tile(SKIN, (len(mu), 1)), yaw=yaw)

    for k, val in enumerate([u, e]):                    # uncertainty, error
        a = axs[4 + k]
        # white rim: the observed region sits at 0 = dark blue under jet, where
        # the default near-black outline would disappear.
        draw(a, mu, F, scalar_rgb(val, CMAP_UE, 0, vmax), yaw=yaw, outline=mask,
             outline_color='white')
        a.text(0.5, 0.035, f'{val[mask].mean():.2f} mm', transform=a.transAxes,
               ha='center', fontsize=8.2, color=INK, fontweight='bold',
               bbox=BADGE_BOX)

    # ---- colorbars --------------------------------------------------------- #
    cax = fig.add_subplot(gs[0:2, 3])
    cb = fig.colorbar(ScalarMappable(Normalize(-dmax, dmax), CMAP_DEV), cax=cax)
    cb.outline.set_visible(False)
    cb.set_label('sample $-$ mean, along normal (mm)', fontsize=6.8, labelpad=3,
                 color=MUTED)
    cb.ax.tick_params(labelsize=6.2, length=2, pad=1.5)

    cax = fig.add_subplot(gs[2, 3])
    cb = fig.colorbar(ScalarMappable(Normalize(0, vmax), CMAP_UE), cax=cax)
    cb.outline.set_visible(False)
    cb.set_label('mm (shared)', fontsize=6.8, labelpad=3, color=MUTED)
    cb.ax.tick_params(labelsize=6.2, length=2, pad=1.5)

    save(fig, outdir, 'fig1_samples')


# --------------------------------------------------------------------------- #
# figures 2-4 - analysis
# --------------------------------------------------------------------------- #
def figure_scatter(rows, outdir):
    fig, ax = plt.subplots(figsize=(4.7, 3.7)); style(ax)
    lim = 0
    for m in rows:
        ax.scatter(m['um'], m['em'], s=5, alpha=0.30, color=m['colour'],
                   linewidths=0, label=f"{m['label']}   $\\rho={m['rho']:.2f}$")
        lim = max(lim, np.percentile(m['um'], 99.5), np.percentile(m['em'], 99.5))
    ax.plot([0, lim], [0, lim], color=INK, lw=1.0, ls='--', alpha=0.6, zorder=0)
    ax.text(lim * 0.98, lim * 0.88, 'perfect\ncalibration', fontsize=7,
            color=MUTED, ha='right', va='top', style='italic')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('predicted uncertainty $u_i$ (mm)')
    ax.set_ylabel('actual error $e_i$ (mm)')
    ax.set_title('Every missing vertex', loc='left', fontweight='medium', pad=8)
    leg = ax.legend(fontsize=7.6, frameon=False, loc='upper left',
                    handletextpad=0.35, borderpad=0.2)
    for h in leg.legend_handles:
        h.set_alpha(1.0); h.set_sizes([18])
    save(fig, outdir, 'fig2_scatter')


def figure_sparsification(rows, outdir):
    fig, ax = plt.subplots(figsize=(4.7, 3.7)); style(ax)
    for m in rows:
        fr, s, o = sparsification(m['um'], m['em'])
        ax.plot(fr, s, color=m['colour'], lw=1.9,
                label=f"{m['label']}   AUSE $={np.trapz(s - o, fr):.3f}$")
        ax.plot(fr, o, color=m['colour'], lw=1.0, ls=':', alpha=0.8)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls='--')
    ax.text(0.99, 1.015, 'random removal', fontsize=7, color=MUTED, ha='right',
            va='bottom', transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.10)
    ax.set_xlabel('fraction of most-uncertain vertices removed')
    ax.set_ylabel('normalised mean error')
    ax.set_title('Drop the vertices $u$ flags, error collapses', loc='left',
                 fontweight='medium', pad=8)
    ax.legend(fontsize=7.6, frameon=False, loc='lower left', handletextpad=0.5,
              borderpad=0.2)
    ax.text(0.035, 0.30, 'dotted = oracle\n(ranked by true error)', fontsize=7,
            color=MUTED, transform=ax.transAxes, style='italic')
    save(fig, outdir, 'fig3_sparsification')


def figure_reliability(rows, outdir):
    fig, ax = plt.subplots(figsize=(4.7, 3.7)); style(ax)
    U = np.concatenate([m['um'] for m in rows])
    E = np.concatenate([m['em'] for m in rows])
    q = np.quantile(U, np.linspace(0, 1, 11))
    pred, act = [], []
    for b in range(10):
        sel = (U >= q[b]) & (U <= q[b + 1] if b == 9 else U < q[b + 1])
        if sel.any():
            pred.append(U[sel].mean()); act.append(E[sel].mean())
    x = np.arange(len(act))
    ax.bar(x - 0.19, pred, 0.38, color='#9AA3AE', label='predicted $u$')
    ax.bar(x + 0.19, act, 0.38, color='#C2410C', label='actual $e$')
    ax.set_xticks(x); ax.set_xticklabels([str(i + 1) for i in x])
    ax.set_xlabel('uncertainty decile  (low $\\rightarrow$ high)')
    ax.set_ylabel('mean magnitude (mm)')
    ax.set_title('Predicted magnitude tracks actual, decile by decile', loc='left',
                 fontweight='medium', pad=8)
    ax.legend(fontsize=7.6, frameon=False, loc='upper left', borderpad=0.2)
    save(fig, outdir, 'fig4_reliability')


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--yaw', type=float, default=-35.0)
    ap.add_argument('--samples', type=int, nargs=3, default=[0, 7, 13],
                    help='which of the K draws to show as deviation maps')
    ap.add_argument('--hero', default=None,
                    help='mesh shown in fig1 (default: first in the npz)')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    F, K, rows = load(args.npz)
    print(f'K={K}, {len(rows)} meshes -> {args.outdir}')

    hero = next((m for m in rows if m['name'] == args.hero), rows[0])
    figure_samples(F, K, hero, args.outdir, args.yaw, args.samples)
    figure_scatter(rows, args.outdir)
    figure_sparsification(rows, args.outdir)
    figure_reliability(rows, args.outdir)

    print()
    for m in rows:
        print(f"  {m['name']}: u={m['um'].mean():.3f}mm  e={m['em'].mean():.3f}mm  "
              f"ratio={m['um'].mean() / m['em'].mean():.3f}  rho={m['rho']:.3f}")


if __name__ == '__main__':
    main()
