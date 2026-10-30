"""Fig. 4 — participant-grouped 5-fold cross-validation schematic. usage: python make_fig4.py <out_dir>"""
import sys, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
INK, INK2, TEST, TRAIN, GLYPH_TR, GLYPH_TE = '#1a1a1a', '#52514e', '#3d3d3a', '#ecebe6', '#9a998f', '#ffffff'
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'pdf.fonttype': 42})

W, H = 7.2, 3.35
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

# participant trial counts per fold (illustrative): each inner list = participants in that fold
FOLDS = [[1, 2, 1, 3], [1, 2, 1, 1, 1], [1, 1, 4, 1], [2, 1, 1, 1, 1], [1, 1, 2, 1, 1]]
X0, CW, GAP, ROWH, RH = 1.05, 0.96, 0.07, 0.43, 0.30
SQ, SGAP, PGAP = 0.072, 0.016, 0.085

def fold_cell(x, y, is_test, parts):
    ax.add_patch(FancyBboxPatch((x, y), CW, RH, boxstyle='round,pad=0,rounding_size=0.035',
                                fc=TEST if is_test else TRAIN, ec='none'))
    total = sum(n * SQ + (n - 1) * SGAP for n in parts) + PGAP * (len(parts) - 1)
    cx = x + (CW - total) / 2; cy = y + RH / 2 - SQ / 2
    for n in parts:
        for k in range(n):
            ax.add_patch(Rectangle((cx + k * (SQ + SGAP), cy), SQ, SQ, fc=GLYPH_TE if is_test else GLYPH_TR, ec='none'))
        if n > 1:   # bar joining one participant's trials
            ax.add_patch(Rectangle((cx, cy + SQ / 2 - 0.006), n * SQ + (n - 1) * SGAP, 0.012, fc=GLYPH_TE if is_test else GLYPH_TR, ec='none'))
        cx += n * SQ + (n - 1) * SGAP + PGAP

ytop = 2.72
for f in range(5):   # column headers
    ax.text(X0 + f * (CW + GAP) + CW / 2, ytop + RH + 0.12, f'Fold {f + 1}', ha='center', va='bottom', fontsize=8, color=INK2)
for it in range(5):
    y = ytop - it * ROWH
    ax.text(X0 - 0.12, y + RH / 2, f'Iteration {it + 1}', ha='right', va='center', fontsize=8, color=INK)
    for f in range(5):
        fold_cell(X0 + f * (CW + GAP), y, f == it, FOLDS[f])
    xr = X0 + 5 * (CW + GAP) + 0.05
    ax.annotate('', xy=(xr + 0.32, y + RH / 2), xytext=(xr, y + RH / 2),
                arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.8, mutation_scale=8))
    ax.text(xr + 0.38, y + RH / 2, f'predictions for\nfold {it + 1} trials', ha='left', va='center', fontsize=7, color=INK2, linespacing=1.1)

# legend row
yl = ytop - 5 * ROWH + 0.05
lx = X0
ax.add_patch(FancyBboxPatch((lx, yl), 0.32, 0.17, boxstyle='round,pad=0,rounding_size=0.03', fc=TRAIN, ec='none'))
ax.text(lx + 0.4, yl + 0.085, 'Training folds: feature scaling and class weights fitted here', va='center', fontsize=7.5, color=INK)
lx2 = lx + 3.7
ax.add_patch(FancyBboxPatch((lx2, yl), 0.32, 0.17, boxstyle='round,pad=0,rounding_size=0.03', fc=TEST, ec='none'))
ax.text(lx2 + 0.4, yl + 0.085, 'Held-out test fold', va='center', fontsize=7.5, color=INK)
yl2 = yl - 0.27
for k in range(3):
    ax.add_patch(Rectangle((lx + 0.02 + k * (SQ + SGAP), yl2 + 0.05), SQ, SQ, fc=GLYPH_TR, ec='none'))
ax.add_patch(Rectangle((lx + 0.02, yl2 + 0.05 + SQ / 2 - 0.006), 3 * SQ + 2 * SGAP, 0.012, fc=GLYPH_TR, ec='none'))
ax.text(lx + 0.4, yl2 + 0.05 + SQ / 2, 'One participant with three trials: all of a participant\'s trials are always assigned to the same fold',
        va='center', fontsize=7.5, color=INK)
ax.text(lx, yl2 - 0.26, 'XGBoost with fixed hyperparameters (100 trees, maximum depth 3, learning rate 0.1); no hyperparameter tuning.\n'
        'Out-of-fold predictions from the 5 iterations are pooled for evaluation.', va='center', fontsize=7.5, color=INK2, linespacing=1.3)

for ext in ('png', 'tif', 'pdf'):
    kw = {'dpi': 600} if ext != 'pdf' else {}
    if ext == 'tif': kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
    fig.savefig(os.path.join(OUT, f'Fig4_cv_design.{ext}'), bbox_inches='tight', pad_inches=0.05, facecolor='white', **kw)
print('saved')
