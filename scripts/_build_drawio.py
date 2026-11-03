"""
Build the paper's two schematic figures as .drawio files (v3).

  docs/Model_archi/model_architecture_v3.drawio   Fig. 1, the pipeline
  docs/Model_archi/objective_v3.drawio            Fig. 2, the training objective

What v3 fixes relative to the hand-edited v2 that shipped in main.tex:

  * the timestep. v2 drew a single edge from the clock into the BOTTLENECK mesh
    (`edge('etk','clock','m_b','FiLM')`), which is where the "misleading" note
    came from. The code adds the timestep embedding inside all seven residual
    blocks -- three encoder, one middle, three decoder. v3 draws a conditioning
    rail with a tap into every block, and expands one block in an inset so the
    mechanism is visible once rather than asserted seven times.
  * the mechanism label. v2 said "(gamma, beta)" and "FiLM". ResBlock.forward
    adds `self.time(t_emb)` and has no scale branch, so this is an additive
    per-channel bias, not FiLM. v3 says what the code does.
  * the objective. v2's box carried the epsilon-prediction loss with a flat
    missing-x3 weight -- the baseline arm, not the method. The objective now has
    its own figure with the geodesic ramp and the normal term.
  * the sampling column. v2 said "predict noise eps" then "estimate x0 (clip
    +-6)". The network predicts x0 directly, and the clip is gated on
    predict=='eps' in GaussianDiffusion.ddim_sample, so it never fires for the
    method. v3 shows the real step order, including recovering eps from x0-hat.
  * training and inference were tangled inside one box labelled DDIM. They are
    now separate panels.

Equations are rasterized with matplotlib mathtext rather than left to draw.io's
MathJax, so what lands in the PDF does not depend on the exporter.

Ids are namespaced (see `nid`): draw.io reserves a handful of bare names, and a
cell id of "concat" -- which v2 used -- is what produced the "d.setId is not a
function" load failure this file has hit before.

Run (3dmesh env, after scripts/render_figure_assets.py):
  python scripts/_build_drawio.py
"""

import base64
import html
import os
import struct

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AST = os.path.join('docs', 'Model_archi', 'assets')
EQD = os.path.join(AST, 'eq')
OUTD = os.path.join('docs', 'Model_archi')

# ---- Palette: Academic Indigo (kept from v2 so the assets still match) ------
INK = '#1f2937'
ACCENT = '#4f46e5'
ACCENT_DK = '#3730a3'
ACCENT_FILL = '#eef0fb'
BORDER = '#cdd2dc'
PANEL_BD = '#d7dbe3'
ARROW = '#8b91a1'
DASH = '#bcc2d0'
RAIL = '#e8703a'          # the conditioning rail, the one warm colour
WARN_FILL = '#fdf3e7'

PANEL = (f'rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={PANEL_BD};'
         f'verticalAlign=top;align=left;spacingLeft=10;spacingTop=4;fontColor={ACCENT_DK};'
         f'fontStyle=1;fontSize=13;arcSize=2;')
BOX = (f'rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={BORDER};'
       f'fontColor={INK};fontSize=10;arcSize=12;')
ACC = (f'rounded=1;whiteSpace=wrap;html=1;fillColor={ACCENT_FILL};strokeColor={ACCENT};'
       f'fontColor={ACCENT_DK};fontSize=10;fontStyle=1;arcSize=12;')
STEP = (f'rounded=1;whiteSpace=wrap;html=1;fillColor=#f7f8fb;strokeColor={BORDER};'
        f'fontColor={INK};fontSize=10;align=left;spacingLeft=8;arcSize=12;')
NOTE = (f'rounded=1;whiteSpace=wrap;html=1;fillColor={WARN_FILL};strokeColor=#e8bd8a;'
        f'fontColor=#7c4a12;fontSize=9;align=left;spacingLeft=8;arcSize=12;')
LBL = (f'text;html=1;strokeColor=none;fillColor=none;align=center;fontColor={INK};'
       f'fontSize=10;fontStyle=1;')
# whiteSpace=wrap is not optional on these: a text cell without it never wraps,
# it just runs off to the right and drags the exported bounding box with it.
CAP = (f'text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=center;'
       f'fontColor=#6b7280;fontSize=9;')
CAPL = (f'text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=left;'
        f'fontColor=#6b7280;fontSize=9;verticalAlign=top;')
TAP = (f'ellipse;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={RAIL};'
       f'fontColor={RAIL};fontSize=11;fontStyle=1;')


# ---------------------------------------------------------------- primitives
class Doc:
    """One .drawio page. Ids are namespaced so no cell can collide with a name
    draw.io treats as reserved."""

    def __init__(self, name, w, h):
        self.name, self.w, self.h = name, w, h
        self.cells = []
        self._n = 0

    def nid(self, stem):
        self._n += 1
        return f'c{self._n}_{stem}'

    def add(self, stem, value, style, x, y, w, h):
        i = self.nid(stem)
        # draw.io only breaks a line on &#10;. Escaping first and substituting
        # after is what keeps the entity intact -- html.escape would turn the
        # ampersand of a pre-written &#10; into &amp;#10; and print it verbatim.
        value = html.escape(value).replace('\n', '&#10;')
        self.cells.append(
            f'<mxCell id="{i}" value="{value}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" '
            f'height="{h}" as="geometry"/></mxCell>')
        return i

    def img(self, stem, path, x, y, w=None, h=None, keep_aspect=True):
        """Place a PNG. Give w or h; the other follows the file's real aspect."""
        iw, ih = png_size(path)
        if w is None and h is None:
            w = iw
        if keep_aspect:
            if h is None:
                h = round(w * ih / iw)
            elif w is None:
                w = round(h * iw / ih)
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        i = self.nid(stem)
        style = (f'shape=image;html=1;imageAspect=0;aspect=fixed;'
                 f'image=data:image/png,{b64};')
        self.cells.append(
            f'<mxCell id="{i}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry"/></mxCell>')
        return i

    def edge(self, src, dst, value='', color=ARROW, dashed=0, arrow='block',
             style_extra='', points=None, exit_=None, entry=None):
        i = self.nid('e')
        st = (f'edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow={arrow};'
              f'strokeColor={color};dashed={dashed};fontSize=9;fontColor=#5b6170;'
              f'labelBackgroundColor=#ffffff;{style_extra}')
        if exit_:
            st += f'exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;'
        if entry:
            st += f'entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;'
        geo = '<mxGeometry relative="1" as="geometry"/>'
        if points:
            pts = ''.join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in points)
            geo = (f'<mxGeometry relative="1" as="geometry">'
                   f'<Array as="points">{pts}</Array></mxGeometry>')
        self.cells.append(
            f'<mxCell id="{i}" value="{html.escape(value)}" style="{st}" edge="1" '
            f'parent="1" source="{src}" target="{dst}">{geo}</mxCell>')
        return i

    def write(self, path):
        xml = (
            '<mxfile host="app.diagrams.net" type="device">'
            f'<diagram name="{self.name}" id="{self.name}">'
            f'<mxGraphModel dx="1400" dy="900" grid="0" gridSize="10" guides="1" '
            f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            f'pageWidth="{self.w}" pageHeight="{self.h}" math="0" shadow="0">'
            '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
            + ''.join(self.cells) +
            '</root></mxGraphModel></diagram></mxfile>')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(xml)
        print(f'wrote {path}  ({len(xml)/1e6:.2f} MB, {len(self.cells)} cells)')


def png_size(path):
    """(width, height) from the PNG IHDR -- avoids a PIL dependency here."""
    with open(path, 'rb') as f:
        head = f.read(24)
    if head[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f'not a PNG: {path}')
    return struct.unpack('>II', head[16:24])


_eq_cache = {}


def eq(name, latex, fontsize=13, color=INK):
    """Rasterize one equation with mathtext; returns its path.

    Rendering here rather than relying on draw.io's math mode keeps the exported
    figure byte-identical regardless of which exporter runs.
    """
    os.makedirs(EQD, exist_ok=True)
    path = os.path.join(EQD, f'{name}.png')
    key = (latex, fontsize, color)
    if _eq_cache.get(path) == key and os.path.exists(path):
        return path
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, latex, fontsize=fontsize, color=color)
    fig.savefig(path, dpi=400, transparent=True, bbox_inches='tight',
                pad_inches=0.02)
    plt.close(fig)
    _eq_cache[path] = key
    return path


def asset(name):
    return os.path.join(AST, name)


# =============================================================== FIGURE 1
def build_pipeline():
    d = Doc('MeshOffsetDiff', 1780, 940)

    # ---------------------------------------------------------- 1. Input
    d.add('p_in', '1. Input', PANEL, 20, 150, 220, 600)
    d.add('l_par', 'Partial mesh', LBL, 30, 178, 200, 16)
    im_par = d.img('i_par', asset('face_partial.png'), 55, 198, w=150)
    d.add('l_msk', 'Known / missing mask', LBL, 30, 388, 200, 16)
    im_msk = d.img('i_msk', asset('face_mask.png'), 55, 408, w=150)
    d.add('leg',
          '■ known (observed)\n■ missing (to complete)\n'
          'dense correspondence with a shared template',
          f'rounded=1;whiteSpace=wrap;html=1;fillColor=#fbfbfd;strokeColor={BORDER};'
          f'fontColor={INK};fontSize=9;align=left;spacingLeft=8;arcSize=12;',
          35, 600, 190, 76)

    # ------------------------------------------- 2. Forward diffusion
    d.add('p_fwd', '2. Forward diffusion', PANEL, 260, 20, 900, 200)
    xs = [286, 452, 618, 784, 950]
    ims = [d.img(f'i_d{k}', asset(f'diff_{k}.png'), x, 52, w=100) for k, x in enumerate(xs)]
    for k in range(4):
        d.edge(ims[k], ims[k + 1])
    d.img('i_eqfwd', eq('fwd',
                        r'$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,'
                        r'\epsilon,\qquad x_0=\Delta/\sigma$'), 300, 172, w=250)
    d.add('c_fwd', 't = 0 → 999   (cosine schedule, T = 1000)\n'
          'the t = 999 state is what sampling starts from',
          CAP, 566, 166, 240, 28)
    d.add('n_sigma',
          'σ ≈ (2.8, 3.3, 3.5)×10⁻³. Raw offsets sit ~300× '
          'below the injected noise, so without the 1/σ rescaling the signal is '
          'swamped. Frames are drawn in coordinates (T + σxₜ); in the '
          'standardized space the last one is unit-variance noise.',
          NOTE, 566, 190, 578, 24)

    # ------------------------------------------- 3. Denoiser
    d.add('p_den', '3. Graph U-Net denoiser', PANEL, 260, 240, 900, 400)
    d.add('t_par', 'fθ(xₜ, c, t) → x̂₀   ·   ≈ 1.14 M parameters   ·   no global '
          'bottleneck: every level keeps per-vertex features', CAPL, 470, 248, 620, 12)

    inp = d.add('b_in', 'in_proj  7 → 64\n[ xₜ (3) | partial xyz (3) | mask (1) ]',
                BOX, 274, 300, 130, 62)

    # U-shape. Encoder blocks hold chs[l]; decoder blocks take the concatenated
    # skip, hence the doubled input width in their labels.
    bw, bh = 96, 40
    enc = [('ResBlock 64\n5023 v', 424, 300),
           ('ResBlock 128\n2511 v', 528, 358),
           ('ResBlock 256\n1255 v', 632, 416)]
    mid = ('ResBlock 256\n627 v', 736, 474)
    dec = [('ResBlock 512→256\n1255 v', 840, 416),
           ('ResBlock 256→128\n2511 v', 944, 358),
           ('ResBlock 128→64\n5023 v', 1048, 300)]

    e_ids = [d.add(f'b_e{k}', t, BOX, x, y, bw, bh) for k, (t, x, y) in enumerate(enc)]
    m_id = d.add('b_mid', mid[0], ACC, mid[1], mid[2], bw, bh)
    d_ids = [d.add(f'b_d{k}', t, BOX, x, y, bw, bh) for k, (t, x, y) in enumerate(dec)]

    out = d.add('b_out', 'out_proj  64 → 3   →   x̂₀',
                ACC, 1000, 250, 152, 36)

    d.edge(inp, e_ids[0])
    for a, b, lab in ((e_ids[0], e_ids[1], 'D↓  64→128'),
                      (e_ids[1], e_ids[2], 'D↓  128→256'),
                      (e_ids[2], m_id, 'D↓  256→256')):
        d.edge(a, b, lab)
    for a, b, lab in ((m_id, d_ids[0], 'U↑  256→256'),
                      (d_ids[0], d_ids[1], 'U↑  256→128'),
                      (d_ids[1], d_ids[2], 'U↑  128→64')):
        d.edge(a, b, lab)
    d.edge(d_ids[2], out, '', ACCENT)

    # Skips carry the concatenation that doubles each decoder block's input.
    for a, b, lab in ((e_ids[0], d_ids[2], 'skip · concat 64+64'),
                      (e_ids[1], d_ids[1], 'concat 128+128'),
                      (e_ids[2], d_ids[0], 'concat 256+256')):
        d.edge(a, b, lab, DASH, 1, 'none')

    # ---- the conditioning rail: one line, a tap into every block ----------
    rail_y = 596
    t_src = d.add('b_t', 't  →  sinusoidal embed  →  MLP',
                  f'rounded=1;whiteSpace=wrap;html=1;fillColor=#fdf1ea;'
                  f'strokeColor={RAIL};fontColor=#7c4a12;fontSize=10;fontStyle=1;'
                  f'arcSize=12;', 274, rail_y - 16, 130, 34)
    taps = []
    for k, bid in enumerate(e_ids + [m_id] + d_ids):
        cx = [424, 528, 632, 736, 840, 944, 1048][k] + bw / 2
        tap = d.add(f'b_t{k}', '⊕', TAP, cx - 9, rail_y - 9, 18, 18)
        taps.append(tap)
        d.edge(tap, bid, '', RAIL, 0, 'none',
               'endFill=0;strokeWidth=1.1;', exit_=(0.5, 0), entry=(0.5, 1))
    d.edge(t_src, taps[0], '', RAIL, 0, 'none', 'strokeWidth=1.6;')
    for a, b in zip(taps, taps[1:]):
        d.edge(a, b, '', RAIL, 0, 'none', 'strokeWidth=1.6;')
    d.add('c_rail',
          'the timestep is added inside every block — all 7, not only the bottleneck',
          f'text;html=1;strokeColor=none;fillColor=none;align=left;fontColor=#7c4a12;'
          f'fontSize=9;fontStyle=2;', 424, rail_y + 14, 480, 12)

    # ------------------------------------------- ResBlock inset
    d.add('p_blk', 'Inside every block', PANEL, 260, 676, 900, 154)
    chain = [('LayerNorm\n+ SiLU', 300), ('FeaStConv', 400), ('⊕  time bias', 500),
             ('LayerNorm\n+ SiLU', 630), ('FeaStConv', 730), ('⊕  residual', 830)]
    ids = []
    for k, (t, x) in enumerate(chain):
        st = (f'rounded=1;whiteSpace=wrap;html=1;fillColor=#fdf1ea;strokeColor={RAIL};'
              f'fontColor=#7c4a12;fontSize=10;fontStyle=1;arcSize=12;'
              if '⊕' in t and 'time' in t else BOX)
        ids.append(d.add(f'b_c{k}', t, st, x, 716, 86, 42))
    for a, b in zip(ids, ids[1:]):
        d.edge(a, b)
    d.edge(ids[0], ids[-1], 'identity (skip proj if width changes)', DASH, 1, 'block',
           'exitX=0.5;exitY=1;entryX=0.5;entryY=1;', points=[(343, 800), (873, 800)])
    d.add('n_blk',
          'The timestep enters as an additive per-channel bias — a single '
          'Linear(t_dim → C) added after the first convolution. There is no '
          'scale branch, so this is not FiLM.',
          NOTE, 930, 716, 220, 62)

    # ------------------------------------------- 4a. Training
    d.add('p_tr', '4a. Training — one step', PANEL, 1180, 20, 300, 300)
    tr = [('sample  t ~ U{0, …, 999},  ε ~ N(0, I)', 56),
          ('xₜ = q_sample(x₀, t, ε)', 100),
          ('x̂₀ = fθ(xₜ, c, t)      one denoising step', 144),
          ('L = L_pos + λₙ L_nrm       (Fig. 2)', 188)]
    tids = [d.add(f'b_tr{k}', t, ACC if k == 3 else STEP, 1194, y, 272, 34)
            for k, (t, y) in enumerate(tr)]
    for a, b in zip(tids, tids[1:]):
        d.edge(a, b)
    d.add('n_tr',
          'No sampling loop at training time: one t per graph per iteration. '
          'Weights are EMA-averaged (decay 0.9995) for evaluation.',
          NOTE, 1194, 240, 272, 50)

    # ------------------------------------------- 4b. Inference
    d.add('p_inf', '4b. Inference — DDIM × 50  (η = 0)',
          PANEL, 1180, 340, 300, 420)
    st0 = d.add('b_i0', 'x_T ~ N(0, I)', BOX, 1194, 378, 272, 28)
    steps = [('(a)  evaluate fθ  →  x̂₀', 418, STEP),
             ('(b)  recover ε̂ from x̂₀ and xₜ', 460, STEP),
             ('(c)  known region ← observed values\n       hard data consistency',
              502, ACC),
             ('(d)  xₜ₋₁ = √āₜ₋₁ x̂₀ '
              '+ √(1−āₜ₋₁) ε̂', 556, STEP),
             ('(e)  re-noise known region  (RePaint)', 606, STEP)]
    sids = [d.add(f'b_i{k+1}', t, s, 1194, y, 272, 44 if '\n' in t else 34)
            for k, (t, y, s) in enumerate(steps)]
    d.edge(st0, sids[0])
    for a, b in zip(sids, sids[1:]):
        d.edge(a, b)
    d.edge(sids[-1], sids[0], '× 50', ACCENT, 1, 'block',
           'exitX=0;exitY=0.5;entryX=0;entryY=0.5;', points=[(1186, 623), (1186, 435)])
    d.add('n_inf',
          'No x̂₀ clipping: the ±6 clamp in ddim_sample is gated on '
          'ε-prediction and never fires here. Under x₀-prediction the '
          'output is already at a scale that does not depend on t.',
          NOTE, 1194, 660, 272, 58)
    d.add('c_inf', 'single feed-forward pass, no test-time optimization',
          CAP, 1194, 726, 272, 12)

    # ------------------------------------------- 5. Output
    d.add('p_out', '5. Output', PANEL, 1500, 150, 260, 600)
    d.add('b_post', 'un-standardize ×σ  →  add template mean T',
          STEP, 1512, 182, 236, 30)
    d.add('l_comp', 'Completed mesh', LBL, 1512, 220, 236, 14)
    im_c = d.img('i_comp', asset('out_completed.png'), 1560, 238, w=140)
    d.add('c_comp', 'observed vertices reproduced exactly\n(measured std over K: 0.000 mm)',
          CAP, 1512, 400, 236, 24)
    d.add('l_div', 'Diverse completions', LBL, 1512, 430, 236, 14)
    for k in range(3):
        d.img(f'i_dv{k}', asset(f'div_{k}.png'), 1520 + k * 78, 448, w=70)
    d.add('l_unc', 'Per-vertex uncertainty', LBL, 1512, 534, 236, 14)
    im_u = d.img('i_unc', asset('out_uncertainty.png'), 1560, 552, w=140)
    d.img('i_ubar', asset('unc_bar.png'), 1540, 712, w=180)

    # ------------------------------------------- cross-panel wiring
    d.edge(im_par, inp, '', ACCENT, 0, 'block', '', exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(im_msk, inp, 'conditioning c', ACCENT, 0, 'block', '',
           exit_=(1, 0.5), entry=(0, 0.9))
    d.edge(out, tids[3], 'training', DASH, 1, 'block', '',
           exit_=(1, 0.2), entry=(0, 0.5))
    d.edge(out, sids[0], 'sampling', ACCENT, 0, 'block', '',
           exit_=(1, 0.8), entry=(0, 0.5))
    d.edge(sids[2], im_c, 'final x̂₀', ACCENT)
    d.edge(im_c, im_u, 'K = 20 runs → per-vertex std', ARROW, 1)

    d.write(os.path.join(OUTD, 'model_architecture_v3.drawio'))


# =============================================================== FIGURE 2
def build_objective():
    d = Doc('MeshOffsetObjective', 1200, 780)

    d.add('p_tot', '', f'rounded=1;whiteSpace=wrap;html=1;fillColor={ACCENT_FILL};'
          f'strokeColor={ACCENT};arcSize=6;', 20, 20, 1160, 60)
    d.img('i_eqtot', eq('total',
                        r'$\mathcal{L}=\mathcal{L}_{\mathrm{pos}}'
                        r'+\lambda_n\,\mathcal{L}_{\mathrm{nrm}},'
                        r'\qquad \lambda_n=0.01$', 15, ACCENT_DK), 380, 34, w=330)

    # ---- (a) geodesic depth ramp -------------------------------------------
    d.add('p_pos', '(a)  Geodesic depth-weighted position', PANEL, 20, 96, 380, 660)
    d.img('i_eqpos', eq('pos',
                        r'$\mathcal{L}_{\mathrm{pos}}=\mathbb{E}\left[\sum_i w_i\,'
                        r'\|\hat{x}_{0,i}-x_{0,i}\|^2\right]$', 12), 40, 128, w=250)
    d.img('i_eqw', eq('wramp',
                      r'$w_i=\lambda_{\mathrm{kn}}+(w_{\max}-\lambda_{\mathrm{kn}})'
                      r'\,d_i/d_{\max}$', 12), 40, 166, w=250)
    d.add('c_pos', 'on missing vertices;  wᵢ = λₖₙ on observed.  '
          'λₖₙ = 1,  w_max = 3.', CAPL, 40, 198, 340, 24)
    d.img('i_ramp', asset('ramp_weights_zoom.png'), 90, 226, w=240)
    d.img('i_rbar', asset('ramp_bar.png'), 110, 472, w=200)
    d.add('c_ramp',
          'dᵢ: exact geodesic to the known boundary ring, by multi-source '
          'Dijkstra on the template edge graph with travel restricted to the hole, '
          'so a path cannot leak around the outside. The ramp is rescaled to mean '
          'w_max over the hole, which isolates its shape from the overall '
          'missing-vs-observed emphasis — so the deepest weights exceed w_max.',
          CAPL, 36, 532, 348, 68)
    d.add('n_pos',
          'The COMA template is 3 components (face + 2 eyeballs); an eyeball hole '
          'has no boundary ring at all, and those vertices are treated as '
          'maximally deep.', NOTE, 34, 606, 352, 46)

    # ---- (b) surface normal -------------------------------------------------
    d.add('p_nrm', '(b)  Surface-normal orientation', PANEL, 416, 96, 380, 660)
    d.img('i_eqnrm', eq('nrm',
                        r'$\mathcal{L}_{\mathrm{nrm}}=\mathbb{E}_t\left[\bar\alpha_t\,'
                        r'\frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}}'
                        r'\left(1-\langle\hat{n}_i,n_i\rangle\right)\right]$', 12),
          436, 128, w=300)
    d.add('c_nrm', 'area-weighted unit vertex normals, differentiated through to '
          'x̂₀', CAPL, 436, 174, 344, 22)
    d.img('i_nerr', asset('nrm_error.png'), 486, 200, w=240)
    d.img('i_nbar', asset('nrm_bar.png'), 506, 446, w=200)
    d.add('c_nrm2',
          'Three choices are deliberate. Normals are taken on un-standardized '
          'coordinates — σ is anisotropic, and normals measured in the '
          'standardized space are rotated by 10.5° on average. The āₜ '
          'weight concentrates the term where x̂₀ is still informative. '
          'And it is 1−cos, never arccos, whose derivative diverges exactly at '
          'zero error.', CAPL, 432, 506, 348, 96)
    d.add('n_nrm', 'Applied uniformly over the hole: the depth ramp stays a property '
          'of the positional term alone.', NOTE, 430, 606, 352, 46)

    # ---- (c) why it is not redundant ---------------------------------------
    d.add('p_why', '(c)  Not a restatement of ℓ²', PANEL, 812, 96, 368, 660)
    d.img('i_rho', asset('rho_strip.png'), 826, 130, w=340)
    d.add('c_why',
          'Within a completion, positional and orientation error are close to '
          'uncorrelated: median Spearman ρ = +0.09 over the 20 test '
          'completions, measured on the arm without the normal term. The '
          'positional loss is nearly blind to orientation, so the term supplies '
          'signal ℓ² does not carry. Circled: the completion shown in '
          'Fig. 1.', CAPL, 828, 244, 340, 92)
    d.add('n_why',
          'Pooling vertices across completions instead gives ρ = +0.56, '
          'because per-sample difficulty is a shared confounder — the '
          'within-completion figure is the one the loss acts on.',
          NOTE, 826, 344, 340, 58)
    d.add('c_why2',
          'What we deliberately do not add: a Laplacian smoothness penalty. '
          'Measured on the completions, predicted-to-ground-truth Laplacian '
          'magnitude is 0.996 — the surface already carries the right amount '
          'of high-frequency content and errs by misplacing it, not by being too '
          'rough. A smoothness term would push roughness below the correct level '
          'while improving its own objective. The error field is 28× rougher '
          'than the surface it sits on, which is what a normal term sees and '
          'ℓ² does not.', CAPL, 828, 414, 340, 150)

    d.write(os.path.join(OUTD, 'objective_v3.drawio'))


if __name__ == '__main__':
    os.makedirs(OUTD, exist_ok=True)
    build_pipeline()
    build_objective()
