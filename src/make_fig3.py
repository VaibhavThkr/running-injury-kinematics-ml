"""Fig. 3 — the 76 discrete variables by joint, plane and variable type, marking the 36 that are 0 in every trial.
usage: python make_fig3.py <data_dir> <out_dir>"""
import os, sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variables import VARIABLES

DATA, OUT = sys.argv[1], sys.argv[2]
run = pd.read_csv(os.path.join(DATA, 'ml_run_dataset.csv')).dropna()
walk = pd.read_csv(os.path.join(DATA, 'ml_walk_dataset.csv')).dropna()
const = {v[0]: bool(run[v[0]].nunique() == 1 and walk[v[0]].nunique() == 1) for v in VARIABLES}
assert sum(const.values()) == 36

ROWS = [('Pelvis', 'drop', 'frontal', ('PELVIS_', 'PEAK_PELVIC')), ('Hip', 'extension', 'sagittal', ('HIP_EXT',)),
        ('Hip', 'adduction', 'frontal', ('HIP_ADD', 'PEAK_HIP_ADD')), ('Hip', 'abduction', 'frontal', ('HIP_ABD',)),
        ('Hip', 'rotation', 'transverse', ('HIP_ROT',)), ('Knee', 'flexion', 'sagittal', ('KNEE_FLEX',)),
        ('Knee', 'adduction', 'frontal', ('KNEE_ADD',)), ('Knee', 'abduction', 'frontal', ('KNEE_ABD',)),
        ('Knee', 'rotation', 'transverse', ('KNEE_ROT',)), ('Ankle', 'dorsiflexion', 'sagittal', ('ANKLE_DF',)),
        ('Ankle', 'eversion', 'frontal', ('ANKLE_EVE',)), ('Ankle', 'rotation', 'transverse', ('ANKLE_ROT',))]
COLS = ['Peak angle', 'Excursion', 'Angle at initial contact', 'Timing of peak angle', 'Peak angular velocity', 'Timing of peak angular velocity']
COL_LAB = ['Peak\nangle', 'Excursion', 'Angle at\ninitial\ncontact', 'Timing of\npeak angle', 'Peak\nangular\nvelocity', 'Timing of\npeak\nvelocity']
grid = {}
for name, region, plane, vtype, desc, unit in VARIABLES:
    if vtype in COLS:
        r = [i for i, row in enumerate(ROWS) if name.startswith(row[3])]
        assert len(r) == 1, name
        assert (r[0], vtype) not in grid, name
        grid[(r[0], COLS.index(vtype))] = name
assert len(grid) == 62
FOOT = [v for v in VARIABLES if v[1] == 'Foot']; SPT = [v for v in VARIABLES if v[1] == 'Spatiotemporal']

INK, INK2, DARK, EDGE_L = '#1a1a1a', '#52514e', '#3d4f63', '#b9b8b3'
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 7.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
W, H = 7.0, 4.25
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
CW, CH, GAP = 0.40, 0.205, 0.04


def cell(x, y, name):
    if const[name]:
        ax.add_patch(Rectangle((x, y), CW, CH, fc='white', ec=EDGE_L, lw=0.7))
        ax.text(x + CW / 2, y + CH / 2, '0', ha='center', va='center', fontsize=6.5, color='#8a8984')
    else:
        ax.add_patch(Rectangle((x, y), CW, CH, fc=DARK, ec=DARK, lw=0.7))


# ---- grid of joint variables ----
X0, YTOP = 1.42, 3.36
ax.text(0.10, H - 0.16, 'Joint and pelvis variables (62)', fontsize=8.5, fontweight='bold', va='center')
for j, lab in enumerate(COL_LAB):
    ax.text(X0 + j * (CW + GAP) + CW / 2, YTOP + 0.06, lab, ha='center', va='bottom', fontsize=6.8, color=INK2, linespacing=1.05)
for i, (seg, motion, plane, _) in enumerate(ROWS):
    y = YTOP - (i + 1) * (CH + 0.035)
    ax.text(0.10, y + CH / 2, f'{seg} {motion}', ha='left', va='center', fontsize=7.5, color=INK)
    ax.text(X0 - 0.06, y + CH / 2, plane, ha='right', va='center', fontsize=6.5, color='#8a8984', style='italic')
    for j in range(len(COLS)):
        if (i, j) in grid: cell(X0 + j * (CW + GAP), y, grid[(i, j)])

# ---- foot and spatiotemporal variables ----
XR = 4.30
def column(title, items, ytop):
    ax.text(XR, ytop, title, fontsize=8.5, fontweight='bold', va='center')
    for k, (name, region, plane, vtype, desc, unit) in enumerate(items):
        y = ytop - 0.30 - k * (CH + 0.035)
        cell(XR, y, name)
        ax.text(XR + CW + 0.08, y + CH / 2, desc.split(' (')[0], ha='left', va='center', fontsize=7.2, color=INK)
    return ytop - 0.30 - len(items) * (CH + 0.035)
yend = column(f'Foot variables ({len(FOOT)})', FOOT, H - 0.16)
column(f'Spatiotemporal variables ({len(SPT)})', SPT, yend - 0.12)

# ---- legend ----
n_var = sum(not c for c in const.values())
ax.add_patch(Rectangle((0.10, 0.14), CW * 0.6, CH * 0.8, fc=DARK, ec=DARK, lw=0.7))
ax.text(0.10 + CW * 0.6 + 0.08, 0.14 + CH * 0.4, f'Varied across trials ({n_var})', va='center', fontsize=7.5)
ax.add_patch(Rectangle((1.75, 0.14), CW * 0.6, CH * 0.8, fc='white', ec=EDGE_L, lw=0.7))
ax.text(1.75 + CW * 0.3, 0.14 + CH * 0.4, '0', ha='center', va='center', fontsize=6.5, color='#8a8984')
ax.text(1.75 + CW * 0.6 + 0.08, 0.14 + CH * 0.4, f'0 in every walking and running trial ({76 - n_var})', va='center', fontsize=7.5)

os.makedirs(OUT, exist_ok=True)
for ext in ('png', 'pdf', 'tif'):
    kw = dict(dpi=600) if ext in ('png', 'tif') else {}
    if ext == 'tif': kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
    fig.savefig(os.path.join(OUT, f'Fig3_variables.{ext}'), facecolor='white', **kw)
print('saved Fig3_variables')
