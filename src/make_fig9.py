"""Fig. 9 - treadmill speed by modality and injury group (split violins, truncated at the observed range, with every trial shown).
usage: python make_fig9.py <data_dir> <out_dir>
"""
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
NAMES = {'PFPS': 'PFPS', 'ITBS': 'ITBS', 'Plantar': 'Plantar fasciitis', 'Achilles': 'Achilles tendinopathy', 'MTSS': 'MTSS'}
ORDER = ['PFPS', 'ITBS', 'Plantar', 'Achilles', 'MTSS']          # display order (as Table 1)
RUN_C, WALK_C, INK, INK2 = '#e34948', '#2a78d6', '#1a1a1a', '#52514e'
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'axes.linewidth': 0.8, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
run = pd.read_csv(os.path.join(DATA, 'ml_run_dataset.csv')).dropna()
walk = pd.read_csv(os.path.join(DATA, 'ml_walk_dataset.csv')).dropna()
rng = np.random.default_rng(42)
HALF = 0.42                                   # maximum half-width of a violin

fig, ax = plt.subplots(figsize=(7.0, 3.5))
for i, inj in enumerate(ORDER):
    for df, side, col in [(run, -1, RUN_C), (walk, +1, WALK_C)]:
        v = df.loc[df['Group'] == inj, 'Speed'].values
        y = np.linspace(v.min(), v.max(), 400)                       # truncated at the observed range
        d = gaussian_kde(v)(y); d = d / d.max() * HALF
        ax.fill_betweenx(y, i, i + side * d, color=col, alpha=0.30, lw=0)
        ax.plot(i + side * d, y, color=col, lw=0.8)
        # every trial, spread within the violin outline
        dv = gaussian_kde(v)(v) / gaussian_kde(v)(y).max() * HALF
        ax.scatter(i + side * rng.uniform(0.04, 0.92, len(v)) * dv, v, s=2.2, color=col, alpha=0.75, lw=0, zorder=3)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        for val, ls in [(q1, (0, (1, 1.2))), (med, '-'), (q3, (0, (1, 1.2)))]:
            w = np.interp(val, y, d)
            ax.plot([i, i + side * w], [val, val], color=INK, lw=0.9 if ls == '-' else 0.8, ls=ls, zorder=4)
    ax.plot([i, i], [0.65, 4.25], color=INK2, lw=0.5, zorder=1)
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([NAMES[k] for k in ORDER])
ax.set_xlim(-0.6, len(ORDER) - 0.4); ax.set_ylim(0.6, 4.3)
ax.set_ylabel('Treadmill speed (m/s)'); ax.tick_params(axis='x', length=0)
fig.legend(handles=[Patch(fc=RUN_C, alpha=0.5, ec=RUN_C, label=f'Running ({len(run)} trials; left half)'),
                   Patch(fc=WALK_C, alpha=0.5, ec=WALK_C, label=f'Walking ({len(walk)} trials; right half)')],
          loc='lower center', fontsize=7.5, ncol=2, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.07, 1, 1))
for ext in ('png', 'tif', 'pdf'):
    kw = {'dpi': 600} if ext != 'pdf' else {}
    if ext == 'tif': kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
    fig.savefig(os.path.join(OUT, f'Fig9_speed.{ext}'), bbox_inches='tight', pad_inches=0.04, facecolor='white', **kw)
print('saved; speed 5th-95th percentiles: running', np.percentile(run.Speed, [5, 95]).round(2), 'walking', np.percentile(walk.Speed, [5, 95]).round(2))
