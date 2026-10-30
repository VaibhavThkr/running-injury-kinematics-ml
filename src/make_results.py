"""Table 2, Supplementary Tables S1a and S1b, Figs 5-8 and Fig 10.
usage: python make_results.py <data_dir> <out_dir>   (needs reanalysis_results.json from reanalysis.py in out_dir)
"""
import sys, os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, accuracy_score, average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reanalysis as R

DATA, OUT = sys.argv[1], sys.argv[2]
FIG = os.path.join(OUT, 'figures'); os.makedirs(FIG, exist_ok=True)
J = json.load(open(os.path.join(OUT, 'reanalysis_results.json')))
CFG = {r['config']: r for r in J['results']}
PRIM, FLAT = CFG['B_grouped_76'], CFG['A_original_flat_76']

NAMES = {'PFPS': 'PFPS', 'ITBS': 'ITBS', 'Plantar': 'Plantar fasciitis', 'Achilles': 'Achilles tendinopathy', 'MTSS': 'MTSS'}
ORDER = ['PFPS', 'ITBS', 'Plantar', 'Achilles', 'MTSS']          # display order (as Table 1)
RUN_C, WALK_C, NEUTRAL, INK, INK2 = '#e34948', '#2a78d6', '#8a8984', '#1a1a1a', '#52514e'
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'axes.linewidth': 0.8, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titlesize': 9, 'axes.titleweight': 'bold', 'legend.frameon': False,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})

def save(fig, name):
    for ext in ('png', 'tif', 'pdf'):
        kw = {'dpi': 600} if ext != 'pdf' else {}
        if ext == 'tif': kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), bbox_inches='tight', pad_inches=0.04, facecolor='white', **kw)
    plt.close(fig)

run, walk = R.load('run'), R.load('walk')
feats = R.features(run)

# ---------------- out-of-fold predictions (primary = grouped, seed 42) ----------------
oof = {}
for mod, df in [('run', run), ('walk', walk)]:
    for inj in ORDER:
        y, p, _ = R.oof_binary(df, feats, inj, grouped=True, seed=42); oof[(mod, inj)] = (y, p)
for (mod, inj), (y, p) in oof.items():   # must match the JSON point estimates exactly
    assert abs(roc_auc_score(y, p) - PRIM['injuries'][inj][f'{mod}_auc']) < 1e-12

# ---------------- Table 2 ----------------
rows = []
for inj in ORDER:
    r = PRIM['injuries'][inj]
    rows.append(dict(Injury=NAMES[inj], Modality='Walking', AUC=f"{r['walk_auc']:.3f} [{r['walk_ci'][0]:.3f}, {r['walk_ci'][1]:.3f}]",
                     Diff=f"{r['diff']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}]".replace('+', '+').replace('-', '−'),
                     p=f"{r['p']:.3f}", p_holm=f"{r['p_holm']:.3f}"))
    rows.append(dict(Injury='', Modality='Running', AUC=f"{r['run_auc']:.3f} [{r['run_ci'][0]:.3f}, {r['run_ci'][1]:.3f}]",
                     Diff='', p='', p_holm=''))
T2 = pd.DataFrame(rows); T2.to_csv(os.path.join(OUT, 'Table2_corrected.csv'), index=False)
print('TABLE 2 (participant-grouped CV, 76 features)'); print(T2.to_string(index=False))

# ---------------- seeds (cached) ----------------
seed_file = os.path.join(OUT, 'seeds_20.json')
if not os.path.exists(seed_file):
    def one(seed, grouped):
        out = {}
        for mod, df in [('run', run), ('walk', walk)]:
            for inj in ORDER:
                y, p, _ = R.oof_binary(df, feats, inj, grouped, seed); out[f'{mod}|{inj}'] = roc_auc_score(y, p)
            y, pred, _ = R.oof_multiclass(df, feats, grouped, seed); out[f'{mod}|acc'] = accuracy_score(y, pred)
        return dict(seed=seed, grouped=grouped, res=out)
    S = Parallel(n_jobs=-1)(delayed(one)(s, g) for s in range(20) for g in (False, True))
    json.dump(S, open(seed_file, 'w'))
S = json.load(open(seed_file))
def seed_range(mod, key, grouped):
    v = [s['res'][f'{mod}|{key}'] for s in S if s['grouped'] == grouped]; return np.median(v), min(v), max(v)

# ---------------- one trial per participant ----------------
one_trial = {}
for mod, df in [('run', run), ('walk', walk)]:
    d1 = df.drop_duplicates('sub_id', keep='first').reset_index(drop=True)
    for inj in ORDER:
        y, p, _ = R.oof_binary(d1, feats, inj, grouped=False, seed=42); one_trial[(mod, inj)] = roc_auc_score(y, p)
    y, pred, _ = R.oof_multiclass(d1, feats, grouped=False, seed=42)
    one_trial[(mod, 'acc')] = accuracy_score(y, pred); one_trial[(mod, 'majority')] = np.bincount(y).max() / len(y)

# ---------------- Supplementary Table S1 (a: CV design and sample; b: feature sets) ----------------
def auc_cell(key, inj, mod): return f"{CFG[key]['injuries'][inj][f'{mod}_auc']:.3f}"
def acc_cell(key, mod): return f"{CFG[key]['multiclass'][mod]['acc']:.3f}"
def diff_cell(key, inj):
    r = CFG[key]['injuries'][inj]; return f"{r['diff']:+.3f} ({r['p_holm']:.3f})".replace('-', '−')

def build_part(cols, extra):
    """cols: [(config key or None, column name)]; extra(name, kind, mod, inj) fills columns whose key is None"""
    rows = []
    for inj in ORDER:
        for mod, lab in [('run', 'Running'), ('walk', 'Walking')]:
            row = {'Injury': NAMES[inj], 'Modality': lab}
            for key, name in cols: row[name] = auc_cell(key, inj, mod) if key else extra(name, 'auc', mod, inj)
            rows.append(row)
    for mod, lab in [('run', 'Running'), ('walk', 'Walking')]:
        row = {'Injury': '5-way accuracy', 'Modality': lab}
        for key, name in cols: row[name] = acc_cell(key, mod) if key else extra(name, 'acc', mod, None)
        rows.append(row)
    for inj in ORDER:
        row = {'Injury': NAMES[inj], 'Modality': 'Walk − run (Holm p)'}
        for key, name in cols: row[name] = diff_cell(key, inj) if key else extra(name, 'diff', None, inj)
        rows.append(row)
    return pd.DataFrame(rows).fillna('')

SEEDS_COL, ONE_COL = '20 fold assignments: median [range]', 'One session per participant'
def cv_extra(name, kind, mod, inj):
    if kind == 'diff': return ''
    if name == SEEDS_COL:
        md, mn, mx = seed_range(mod, inj if kind == 'auc' else 'acc', True); return f"{md:.3f} [{mn:.3f}, {mx:.3f}]"
    if name == ONE_COL:
        if kind == 'auc': return f"{one_trial[(mod, inj)]:.3f}"
        return f"{one_trial[(mod, 'acc')]:.3f} (majority {one_trial[(mod, 'majority')]:.3f})"
S1a = build_part([('A_original_flat_76', 'Trial-level CV'), ('B_grouped_76', 'Participant-grouped CV (primary)'),
                  (None, SEEDS_COL), (None, ONE_COL), ('E_grouped_76_paired_subset_323', 'Participants with both modalities (n = 323)')], cv_extra)
DEMO_KEY = 'I_grouped_age_mass_height_only'
def fs_extra(name, kind, mod, inj):
    return '—' if kind == 'diff' else (auc_cell(DEMO_KEY, inj, mod) if kind == 'auc' else acc_cell(DEMO_KEY, mod))
S1b = build_part([('B_grouped_76', 'All 76 variables (primary)'), ('C_grouped_74_no_stride_length_rate', 'Without stride length and rate'),
                  ('D_grouped_70_no_spatiotemporal', 'Without 6 spatiotemporal variables'), ('F_grouped_76_speed_residualized', 'Speed-residualized'),
                  ('G_grouped_77_plus_speed', 'With speed'), ('H_grouped_80_plus_age_mass_height_speed', 'With speed, age, mass and height'),
                  (None, 'Age, mass and height only')], fs_extra)
S1a.to_csv(os.path.join(OUT, 'TableS1a_cv_design_and_sample.csv'), index=False)
S1b.to_csv(os.path.join(OUT, 'TableS1b_feature_sets.csv'), index=False)
print('\nTABLE S1a'); print(S1a.to_string(index=False)); print('\nTABLE S1b'); print(S1b.to_string(index=False))

# ---------------- Fig 5: ROC small multiples ----------------
def small_multiples(kind):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    for ax, inj in zip(axes.flat, ORDER):
        r = PRIM['injuries'][inj]
        for mod, col, ls, lab in [('run', RUN_C, '-', 'Running'), ('walk', WALK_C, '--', 'Walking')]:
            y, p = oof[(mod, inj)]
            if kind == 'roc':
                fpr, tpr, _ = roc_curve(y, p); ax.plot(fpr, tpr, color=col, lw=1.6, ls=ls)
                txt = f"{lab}: {r[f'{mod}_auc']:.3f} [{r[f'{mod}_ci'][0]:.3f}, {r[f'{mod}_ci'][1]:.3f}]"
            else:
                pr, rc, _ = precision_recall_curve(y, p); ax.step(rc, pr, where='post', color=col, lw=1.4, ls=ls)
                ax.axhline(y.mean(), color=col, lw=0.8, ls=':', alpha=0.9)
                txt = f"{lab}: {r[f'{mod}_ap']:.3f} ({r[f'{mod}_prev']:.3f})"
            ax.text(0.98, (0.15 if mod == 'run' else 0.05) if kind == 'roc' else (0.72 if mod == 'run' else 0.63),
                    txt, transform=ax.transAxes, ha='right', va='center', fontsize=6.8, color=INK, zorder=5,
                    bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.92))
            ax.plot([], [], color=col, ls=ls)  # keep handles simple
        if kind == 'roc':
            ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=0.8, ls=(0, (2, 2)))
            ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
        else:
            ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_ylim(0, 1.02)
        ax.set_xlim(0, 1); ax.set_title(NAMES[inj], loc='left')
        ax.set_aspect('equal' if kind == 'roc' else 'auto')
    leg_ax = axes.flat[5]; leg_ax.axis('off')
    handles = [Line2D([], [], color=RUN_C, lw=1.6, ls='-', label='Running'), Line2D([], [], color=WALK_C, lw=1.6, ls='--', label='Walking')]
    if kind == 'roc':
        handles.append(Line2D([], [], color=NEUTRAL, lw=0.8, ls=(0, (2, 2)), label='Chance (AUC = 0.5)'))
        note = ('One-versus-rest ROC curves from\nout-of-fold predictions under\nparticipant-grouped 5-fold\ncross-validation. Values are\nAUC [95% CI] (participant-level\nbootstrap).')
    else:
        handles.append(Line2D([], [], color=NEUTRAL, lw=0.8, ls=':', label='Class prevalence'))
        note = ('Precision-recall curves from the\nsame out-of-fold predictions.\nValues are average precision\n(class prevalence among trials,\nthe precision of a random\nclassifier; dotted lines).')
    leg_ax.legend(handles=handles, loc='upper left', fontsize=8, handlelength=2.6)
    leg_ax.text(0.0, 0.52, note, transform=leg_ax.transAxes, fontsize=7.5, color=INK2, va='top')
    fig.tight_layout(w_pad=1.2, h_pad=1.4)
    return fig
save(small_multiples('roc'), 'Fig5_ROC_grouped')
save(small_multiples('pr'), 'Fig6_PR_grouped')

# ---------------- Fig 7: walk - run differences ----------------
fig, ax = plt.subplots(figsize=(5.2, 2.6))
for i, inj in enumerate(ORDER[::-1]):
    r = PRIM['injuries'][inj]
    ax.plot(r['diff_ci'], [i, i], color=INK2, lw=1.4, solid_capstyle='butt')
    ax.plot([r['diff_ci'][0]] * 2, [i - 0.12, i + 0.12], color=INK2, lw=1.0); ax.plot([r['diff_ci'][1]] * 2, [i - 0.12, i + 0.12], color=INK2, lw=1.0)
    ax.plot(r['diff'], i, 'o', ms=5.5, color=INK, mec='white', mew=1.0, zorder=3)
    ax.text(0.205, i, f"{r['diff']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}]   p = {r['p']:.3f}; Holm p = {r['p_holm']:.3f}".replace('-', '−'),
            va='center', ha='left', fontsize=7, color=INK)
ax.axvline(0, color=NEUTRAL, lw=0.9, ls=(0, (3, 2)))
ax.set_yticks(range(5)); ax.set_yticklabels([NAMES[i] for i in ORDER[::-1]])
ax.set_xlim(-0.2, 0.2); ax.set_xlabel('AUC difference, walking − running (95% CI)')
ax.text(-0.195, 4.75, '← running higher', fontsize=7, color=INK2, va='bottom'); ax.text(0.195, 4.75, 'walking higher →', fontsize=7, color=INK2, va='bottom', ha='right')
ax.set_ylim(-0.6, 5.1)
save(fig, 'Fig7_difference_grouped')

# ---------------- Fig 8: confusion matrices ----------------
CLS = ['PFPS', 'ITBS', 'Achilles', 'Plantar', 'MTSS']; CLSN = ['PFPS', 'ITBS', 'Achilles\ntendinopathy', 'Plantar\nfasciitis', 'MTSS']
disp = [0, 1, 3, 2, 4]   # show in Table 1 order: PFPS, ITBS, plantar, Achilles, MTSS
fig = plt.figure(figsize=(7.4, 3.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.55)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]; cax = fig.add_subplot(gs[0, 2])
for ax, mod, lab in zip(axes, ['run', 'walk'], ['Running', 'Walking']):
    cm = np.array(PRIM['multiclass'][mod]['cm'])[np.ix_(disp, disp)]
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(pct, cmap='Blues', vmin=0, vmax=70)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{pct[i, j]:.1f}", ha='center', va='center', fontsize=7.5, color='white' if pct[i, j] > 40 else INK)
    names = [CLSN[k] for k in disp]
    ax.set_xticks(range(5)); ax.set_xticklabels([n.replace('\n', ' ') for n in names], fontsize=7, rotation=35, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(5)); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Predicted injury'); ax.set_ylabel('True injury' if mod == 'run' else '')
    m = PRIM['multiclass'][mod]
    ax.set_title(f"{lab}\naccuracy {m['acc']*100:.1f}%; majority class (PFPS) {m['majority']*100:.1f}%", loc='left', fontsize=8.5)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
cb = fig.colorbar(im, cax=cax); cb.set_label('% of true-class trials', fontsize=7.5); cb.outline.set_visible(False)
save(fig, 'Fig8_confusion_grouped')

# ---------------- Fig 10: effect of CV design ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, mod, lab, col in zip(axes, ['run', 'walk'], ['Running', 'Walking'], [RUN_C, WALK_C]):
    for i, inj in enumerate(ORDER[::-1]):
        f42 = FLAT['injuries'][inj][f'{mod}_auc']; g42 = PRIM['injuries'][inj][f'{mod}_auc']
        _, fmn, fmx = seed_range(mod, inj, False); _, gmn, gmx = seed_range(mod, inj, True)
        ax.plot([fmn, fmx], [i + 0.13] * 2, color=NEUTRAL, lw=2.2, solid_capstyle='round', alpha=0.55)
        ax.plot([gmn, gmx], [i - 0.13] * 2, color=col, lw=2.2, solid_capstyle='round', alpha=0.35)
        ax.plot(f42, i + 0.13, 'o', ms=6, mfc='white', mec=INK2, mew=1.1, zorder=3)
        ax.plot(g42, i - 0.13, 'o', ms=6, color=col, mec='white', mew=0.8, zorder=3)
    ax.axvline(0.5, color=NEUTRAL, lw=0.9, ls=(0, (3, 2)))
    ax.set_xlim(0.38, 0.78); ax.set_title(lab, loc='left'); ax.set_xlabel('One-versus-rest AUC')
    ax.set_yticks(range(5)); ax.set_yticklabels([NAMES[i] for i in ORDER[::-1]])
handles = [Line2D([], [], marker='o', ls='', mfc='white', mec=INK2, ms=6, label='Trial-level CV (a participant\'s sessions can fall in training and test folds)'),
           Line2D([], [], marker='o', ls='', color=INK2, ms=6, label='Participant-grouped CV'),
           Line2D([], [], color=NEUTRAL, lw=2.2, alpha=0.6, label='Range over 20 random fold assignments')]
fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.13, 1, 1))
save(fig, 'Fig10_cv_design')

# ---------------- numbers for the text ----------------
print('\nTEXT NUMBERS')
for mod in ['run', 'walk']:
    m = PRIM['multiclass'][mod]; cm = np.array(m['cm'])
    off = {CLS[k]: {CLS[j]: int(cm[k, j]) for j in range(5) if j != k} for k in range(5)}
    print(mod, 'acc', round(m['acc'], 3), m['acc_ci'], 'bal', round(m['bal_acc'], 3), m['bal_ci'], 'f1', round(m['macro_f1'], 3), m['f1_ci'],
          'majority', round(m['majority'], 3), 'recall', {k: round(v, 3) for k, v in m['recall'].items()})
    for k in ['Achilles', 'Plantar', 'MTSS']:
        tot = sum(off[k].values()) + int(cm[CLS.index(k), CLS.index(k)])
        print('   ', k, 'n', tot, 'predicted as', off[k], 'share PFPS+ITBS of all', round((off[k]['PFPS'] + off[k]['ITBS']) / tot, 3))
for inj in ORDER:
    r = PRIM['injuries'][inj]
    print(inj, 'AP run/walk', round(r['run_ap'], 3), round(r['walk_ap'], 3), 'prev', round(r['run_prev'], 3), round(r['walk_prev'], 3))
for mod in ['run', 'walk']:
    print('flat', mod, 'AUC range', min(FLAT['injuries'][i][f'{mod}_auc'] for i in ORDER), max(FLAT['injuries'][i][f'{mod}_auc'] for i in ORDER),
          'acc', FLAT['multiclass'][mod]['acc'])
print('grouped seed medians < flat seed medians for all:', all(seed_range(m, i, True)[0] < seed_range(m, i, False)[0] for m in ['run', 'walk'] for i in ORDER))
print('grouped seed MAX < flat seed MIN count:', sum(seed_range(m, i, True)[2] < seed_range(m, i, False)[1] for m in ['run', 'walk'] for i in ORDER), 'of 10')
print('one-trial AUC range', min(v for k, v in one_trial.items() if k[1] in ORDER), max(v for k, v in one_trial.items() if k[1] in ORDER))
