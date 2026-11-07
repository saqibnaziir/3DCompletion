"""
Render architecture / flow figures for the diffusion mesh-completion model.

Outputs (PNG @200dpi + SVG vector) into docs/figures/:
  - model_architecture.{png,svg}  : the GraphUNetDenoiser U-Net
  - pipeline_flow.{png,svg}        : diffusion training + DDIM sampling flow
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join('docs', 'figures')
os.makedirs(OUT, exist_ok=True)

# palette
C_IO   = '#e8eef7'
C_ENC  = '#cfe2f3'
C_BOT  = '#fce5cd'
C_DEC  = '#d9ead3'
C_TIME = '#f4cccc'
C_PROC = '#f0f0f0'
C_HL   = '#fff2cc'


def box(ax, cx, cy, w, h, text, fc, ec='#333333', fs=10, tc='black', lw=1.3):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle='round,pad=0.15,rounding_size=0.6',
                 fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color=tc, weight='bold', zorder=3)
    return (cx, cy, w, h)


def arrow(ax, p_from, p_to, color='#333333', ls='-', lw=1.6, style='-|>',
          shrinkA=13, shrinkB=13):
    ax.annotate('', xy=p_to, xytext=p_from,
                arrowprops=dict(arrowstyle=style, color=color, ls=ls, lw=lw,
                                shrinkA=shrinkA, shrinkB=shrinkB,
                                connectionstyle='arc3'), zorder=1)


def label(ax, x, y, text, fs=9, color='#444444', style='italic'):
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            color=color, style=style, zorder=4)


# =====================================================================
# Figure 1 — Denoiser U-Net architecture
# =====================================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    ax.set_title('GraphUNetDenoiser  —  $\\epsilon_\\theta(\\Delta_t,\\ c,\\ t)$   '
                 '(fixed-topology graph U-Net, ~1.14M params)',
                 fontsize=14, weight='bold', pad=12)

    # ---- inputs (left) ----
    box(ax, 11, 90, 20, 6, 'Noisy offset  $\\Delta_t$\n(N×3)', C_IO, fs=10)
    box(ax, 11, 81, 20, 6, 'Partial xyz\n(N×3)', C_IO, fs=10)
    box(ax, 11, 72, 20, 6, 'Mask\n(N×1)', C_IO, fs=10)
    box(ax, 11, 62, 20, 5.5, 'concat  →  (N×7)', C_PROC, fs=10)
    inproj = box(ax, 11, 52, 20, 5.5, 'in_proj  Linear 7→64', C_PROC, fs=10)
    for yy in (90, 81, 72):
        arrow(ax, (21, yy), (11, 64.7), shrinkB=4, lw=1.2)
    arrow(ax, (11, 59.3), (11, 54.7))

    # ---- encoder (down) ----
    eL0 = box(ax, 33, 78, 19, 7, 'ResBlock\n5023 × 64', C_ENC)
    eL1 = box(ax, 33, 58, 19, 7, 'ResBlock\n2511 × 128', C_ENC)
    eL2 = box(ax, 33, 38, 19, 7, 'ResBlock\n1255 × 256', C_ENC)
    arrow(ax, (11, 49.3), (33, 78), shrinkB=14)
    arrow(ax, (33, 74.5), (33, 61.5)); label(ax, 39.5, 68, '↓ D₀  (×2)')
    arrow(ax, (33, 54.5), (33, 41.5)); label(ax, 39.5, 48, '↓ D₁  (×2)')

    # ---- bottleneck ----
    bot = box(ax, 50, 20, 22, 7, 'Bottleneck ResBlock\n627 × 256', C_BOT)
    arrow(ax, (33, 34.5), (50, 23.2)); label(ax, 38, 27, '↓ D₂')

    # ---- decoder (up) ----
    dL2 = box(ax, 67, 38, 19, 7, 'ResBlock\n1255 × 256', C_DEC)
    dL1 = box(ax, 67, 58, 19, 7, 'ResBlock\n2511 × 128', C_DEC)
    dL0 = box(ax, 67, 78, 19, 7, 'ResBlock\n5023 × 64', C_DEC)
    arrow(ax, (50, 23.2), (67, 34.5)); label(ax, 62, 27, '↑ U₂')
    arrow(ax, (67, 41.5), (67, 54.5)); label(ax, 73.5, 48, '↑ U₁')
    arrow(ax, (67, 61.5), (67, 74.5)); label(ax, 73.5, 68, '↑ U₀')

    # ---- skip connections ----
    for (e, d, yy) in [(eL0, dL0, 78), (eL1, dL1, 58), (eL2, dL2, 38)]:
        arrow(ax, (42.5, yy), (57.5, yy), color='#b45f06', ls=(0, (5, 3)), lw=1.6)
    label(ax, 50, 81.5, 'skip ⊕ (concat)', color='#b45f06', fs=9)

    # ---- output ----
    outp = box(ax, 89, 78, 17, 6, 'out_proj\n64→3', C_PROC, fs=10)
    box(ax, 89, 67, 17, 6, '$\\hat{\\epsilon}$  predicted\nnoise (N×3)', C_IO, fs=10)
    arrow(ax, (76.5, 78), (89, 78), shrinkB=12)
    arrow(ax, (89, 75), (89, 70))

    # ---- timestep embedding ----
    box(ax, 12, 25, 22, 7, 'timestep  t\nsinusoidal → MLP', C_TIME, fs=10)
    arrow(ax, (23, 25), (39, 21), color='#cc0000', ls=(0, (4, 3)), lw=1.4, shrinkB=14)
    label(ax, 30, 13, '+ added inside every ResBlock (FiLM)', color='#cc0000', fs=9)

    # ---- ResBlock detail inset ----
    box(ax, 50, 92, 70, 6,
        'ResBlock:  LayerNorm → FeaStConv → (+ t-embed) → LayerNorm → FeaStConv → (+ residual),  SiLU',
        C_HL, fs=9.5)

    # ---- legend ----
    leg = [('Encoder', C_ENC), ('Bottleneck', C_BOT), ('Decoder', C_DEC),
           ('Time', C_TIME), ('I/O', C_IO)]
    for i, (t, c) in enumerate(leg):
        x0 = 6 + i * 11
        ax.add_patch(FancyBboxPatch((x0, 2.5), 2.2, 2.2,
                     boxstyle='round,pad=0.1', fc=c, ec='#333'))
        ax.text(x0 + 3, 3.6, t, fontsize=9, va='center')

    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(OUT, f'model_architecture.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', os.path.join(OUT, 'model_architecture.png/.svg'))


# =====================================================================
# Figure 2 — Diffusion training + DDIM sampling flow
# =====================================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    ax.set_title('Diffusion mesh completion — training and sampling pipeline',
                 fontsize=14, weight='bold', pad=10)

    # ---------- TRAINING (top) ----------
    ax.text(2, 92, 'TRAINING', fontsize=12, weight='bold', color='#1f4e79')
    y = 80
    t1 = box(ax, 11, y, 17, 8, 'Complete mesh\nV  (5023×3)', C_IO, fs=9)
    t2 = box(ax, 30, y, 16, 8, 'normalize\nΔ = V − mean', C_PROC, fs=9)
    t3 = box(ax, 48, y, 16, 8, 'standardize\n$x_0=\\Delta/\\sigma$', C_HL, fs=9)
    t4 = box(ax, 67, y, 18, 8, 'add noise (t)\n$x_t=\\sqrt{\\bar\\alpha}x_0+\\sqrt{1-\\bar\\alpha}\\,\\epsilon$', C_PROC, fs=8.5)
    for a, b in [(t1, t2), (t2, t3), (t3, t4)]:
        arrow(ax, (a[0] + a[2] / 2, y), (b[0] - b[2] / 2, y))

    cnd = box(ax, 67, 62, 18, 7, 'condition  c\n[partial xyz/σ, mask]', C_IO, fs=8.5)
    den = box(ax, 88, 71, 18, 9, 'Denoiser\nU-Net  $\\epsilon_\\theta$', C_ENC, fs=10)
    arrow(ax, (76, 80), (88, 73.5))
    arrow(ax, (76, 62), (88, 68.5))
    loss = box(ax, 88, 54, 20, 7, 'weighted MSE\n$\\|\\hat\\epsilon-\\epsilon\\|^2$ (missing×3)', C_BOT, fs=8.5)
    arrow(ax, (88, 66.5), (88, 57.5)); label(ax, 95, 62, '$\\hat\\epsilon$', fs=11)

    # divider
    ax.plot([0, 100], [46, 46], color='#cccccc', lw=1, ls='--')

    # ---------- SAMPLING (bottom) ----------
    ax.text(2, 40, 'SAMPLING (inference)', fontsize=12, weight='bold', color='#274e13')
    y2 = 28
    s1 = box(ax, 11, y2, 16, 8, 'Partial mesh\n(missing zeroed)', C_IO, fs=9)
    s2 = box(ax, 29, y2, 16, 8, 'condition c\n[xyz/σ, mask]', C_IO, fs=9)
    s3 = box(ax, 46, y2, 13, 8, '$x_T$ = noise', C_PROC, fs=9)
    loop = box(ax, 70, y2, 26, 11,
               'DDIM loop  ×50:\n$\\hat\\epsilon=\\epsilon_\\theta(x_t,c,t)$ →  $\\hat{x}_0$\n'
               'clamp $\\hat{x}_0$ (±6)\nclamp KNOWN = exact (RePaint)', C_DEC, fs=8.5)
    for a, b in [(s1, s2), (s2, s3)]:
        arrow(ax, (a[0] + a[2] / 2, y2), (b[0] - b[2] / 2, y2))
    arrow(ax, (52.5, y2), (57, y2))
    # loop-back arrow
    ax.annotate('', xy=(64, 34.5), xytext=(76, 34.5),
                arrowprops=dict(arrowstyle='-|>', color='#274e13', lw=1.4,
                                connectionstyle='arc3,rad=-0.6'))
    label(ax, 70, 37.5, 't → t−1', color='#274e13', fs=8)

    o1 = box(ax, 70, 11, 16, 7, '$x_0$  (offset)', C_PROC, fs=9)
    o2 = box(ax, 89, 11, 18, 7, '×σ, +mean\nde-standardize', C_PROC, fs=8.5)
    o3 = box(ax, 89, 22, 18, 7, 'blend: known exact,\nmissing = prediction', C_HL, fs=8.5)
    arrow(ax, (70, 22.5), (70, 14.5))
    arrow(ax, (78, 11), (80, 11))
    arrow(ax, (89, 14.5), (89, 18.5))
    fin = box(ax, 89, 33, 18, 7, 'Completed mesh\n(no post-proc)', C_BOT, fs=9)
    arrow(ax, (89, 25.5), (89, 29.5))

    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(OUT, f'pipeline_flow.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', os.path.join(OUT, 'pipeline_flow.png/.svg'))


if __name__ == '__main__':
    fig_architecture()
    fig_pipeline()
    print('done ->', OUT)
