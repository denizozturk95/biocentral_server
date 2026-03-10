import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'figure.dpi': 300,
})

LAYERS = [
    ('#1D6A4A', '#D5F5E3', 'API / Service Layer'),
    ('#7D4E00', '#FDEBD0', 'Inference & Task Infrastructure'),
    ('#5B2C6F', '#F4ECF7', 'Storage & Caching'),
]
ARROW    = '#555555'
CARD_EDGE = '#CCCCCC'

# ── Canvas ─────────────────────────────────────────────────────────────────────
W, H = 9.0, 6.2
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Helpers ────────────────────────────────────────────────────────────────────
def layer_box(ax, x, y, w, h, hdr_col, body_col, title, hdr_h=0.38):
    r = 0.22
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        lw=1.0, edgecolor='#AAAAAA', facecolor=body_col, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + h - hdr_h), w, hdr_h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        lw=0, edgecolor='none', facecolor=hdr_col, zorder=3))
    ax.add_patch(mpatches.Rectangle((x, y + h - hdr_h), w, hdr_h / 2,
        lw=0, edgecolor='none', facecolor=hdr_col, zorder=3))
    ax.text(x + w / 2, y + h - hdr_h / 2, title,
            ha='center', va='center', fontsize=8.5,
            fontweight='bold', color='white', zorder=4)

def card(ax, x, y, w, h, bg, title, sub=None, fs=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.14",
        lw=0.8, edgecolor=CARD_EDGE, facecolor=bg, zorder=5))
    ty = y + h / 2 + (0.12 if sub else 0)
    ax.text(x + w / 2, ty, title,
            ha='center', va='center', fontsize=fs,
            fontweight='bold', color='#1A1A1A', zorder=6)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.17, sub,
                ha='center', va='center', fontsize=6.2,
                color='#555555', style='italic', zorder=6)

def arr_v(ax, x, y0, y1):
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='->', color=ARROW,
                                lw=0.9, mutation_scale=10), zorder=7)

def arr_h(ax, x0, x1, y):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='->', color=ARROW,
                                lw=0.8, mutation_scale=9), zorder=7)

# ── Layout constants ───────────────────────────────────────────────────────────
LX, LW = 0.25, 8.50          # layer left edge and width
PAD = 0.28                    # inner horizontal padding

# ── LAYER 1 – API  ────────────────────────────────────────────────────────────
# 5 endpoint cards in one row
API_Y  = 4.30
API_H  = 1.55
layer_box(ax, LX, API_Y, LW, API_H, LAYERS[0][0], LAYERS[0][1], LAYERS[0][2])

endpoints = ['embed', 'project', 'train', 'predict', 'inference']
n = len(endpoints)
avail = LW - 2 * PAD
gap   = 0.22
cw    = (avail - (n - 1) * gap) / n
ch    = 0.72
cy    = API_Y + (API_H - 0.38) / 2 - ch / 2 + 0.05   # vertically centred below header

for i, lbl in enumerate(endpoints):
    cx = LX + PAD + i * (cw + gap)
    card(ax, cx, cy, cw, ch, '#A9DFBF', lbl, fs=8.2)

# FastAPI badge (top-right corner inside header)
bw, bh = 1.45, 0.30
ax.add_patch(FancyBboxPatch((LX + LW - bw - 0.12, API_Y + API_H - bh - 0.04), bw, bh,
    boxstyle="round,pad=0,rounding_size=0.08",
    lw=0.8, edgecolor='#AAFFAA', facecolor='#0E4D30', zorder=6))
ax.text(LX + LW - bw / 2 - 0.12, API_Y + API_H - bh / 2 - 0.04,
        'FastAPI Gateway',
        ha='center', va='center', fontsize=6.5,
        fontweight='bold', color='#CCFFCC', zorder=7)

# ── LAYER 2 – Infrastructure ──────────────────────────────────────────────────
INF_H  = 1.55
INF_Y  = API_Y - 0.45 - INF_H
layer_box(ax, LX, INF_Y, LW, INF_H, LAYERS[1][0], LAYERS[1][1], LAYERS[1][2])

inf_cards = [
    ('NVIDIA Triton',     'Inference Server', '#FAD7A0'),
    ('Task Workers',      'Async Pipeline',   '#FAD7A0'),
    ('PLM Models',        'ESM2 · ProtT5',    '#AED6F1'),
    ('ONNX / TorchScript','Optimized weights','#AED6F1'),
]
n_inf = len(inf_cards)
cw_inf = (avail - (n_inf - 1) * gap) / n_inf
ch_inf = 0.78
cy_inf = INF_Y + (INF_H - 0.38) / 2 - ch_inf / 2 + 0.05

inf_cx = []
for i, (title, sub, bg) in enumerate(inf_cards):
    cx = LX + PAD + i * (cw_inf + gap)
    inf_cx.append(cx + cw_inf / 2)
    card(ax, cx, cy_inf, cw_inf, ch_inf, bg, title, sub, fs=7.8)

# horizontal connectors inside infra
mid_inf = cy_inf + ch_inf / 2
for i in range(len(inf_cards) - 1):
    x0 = LX + PAD + i * (cw_inf + gap) + cw_inf
    x1 = x0 + gap
    arr_h(ax, x0, x1, mid_inf)

# ── LAYER 3 – Storage ─────────────────────────────────────────────────────────
STO_H  = 1.55
STO_Y  = INF_Y - 0.45 - STO_H
layer_box(ax, LX, STO_Y, LW, STO_H, LAYERS[2][0], LAYERS[2][1], LAYERS[2][2])

sto_cards = [
    ('PostgreSQL',   'Embedding cache',   '#D7BDE2'),
    ('Redis',        'Task queue',         '#D7BDE2'),
    ('File System',  'Model artefacts',    '#D7BDE2'),
    ('Object Store', 'Datasets / results', '#D7BDE2'),
]
n_sto = len(sto_cards)
cw_sto = (avail - (n_sto - 1) * gap) / n_sto
ch_sto = 0.78
cy_sto = STO_Y + (STO_H - 0.38) / 2 - ch_sto / 2 + 0.05

sto_cx = []
for i, (title, sub, bg) in enumerate(sto_cards):
    cx = LX + PAD + i * (cw_sto + gap)
    sto_cx.append(cx + cw_sto / 2)
    card(ax, cx, cy_sto, cw_sto, ch_sto, bg, title, sub, fs=7.8)

# ── VERTICAL ARROWS between layers ────────────────────────────────────────────
# API → Infra: arrows from each endpoint card down
api_cx = [LX + PAD + i * (cw + gap) + cw / 2 for i in range(n)]
# Map 5 endpoints → 4 infra cards (embed→Triton, project→Workers, train→Workers,
#                                   predict→PLM, inference→ONNX)
mapping = [0, 1, 1, 2, 3]
for i, ep_x in enumerate(api_cx):
    arr_v(ax, ep_x, API_Y, API_Y - 0.45)

# Infra → Storage: align by index
for i, ix in enumerate(inf_cx):
    arr_v(ax, ix, INF_Y, INF_Y - 0.45)

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(W / 2, H - 0.18, 'Biocentral Server — System Architecture',
        ha='center', va='top', fontsize=11,
        fontweight='bold', color='#1A1A1A')

plt.tight_layout(pad=0.3)
plt.savefig('/mnt/user-data/outputs/biocentral_architecture.pdf',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/mnt/user-data/outputs/biocentral_architecture.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Done")