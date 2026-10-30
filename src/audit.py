"""Methodological checks of the corrected analysis. usage: python audit.py <data_dir> <out_dir>  (after reanalysis.py)
A. AUC via Mann-Whitney formula vs sklearn      B. Holm vs statsmodels      C. fold-mean vs pooled AUC
D. other models (L2 logistic regression, random forest) under the same participant-grouped CV
E. participant-level scoring (mean prediction per participant)
(The label-permutation test reported in the paper is perm_test.py.)
"""
import sys, os, json, numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reanalysis as R

DATA, OUT = sys.argv[1], sys.argv[2]
J = json.load(open(os.path.join(OUT, 'reanalysis_results.json')))
PRIM = {r['config']: r for r in J['results']}['B_grouped_76']
run, walk = R.load('run'), R.load('walk'); feats = R.features(run)
MODS = [('run', run), ('walk', walk)]
res = {}

def mw_auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

# A + C + E
A, C, E = {}, {}, {}
for mod, df in MODS:
    for inj in R.INJ:
        y, p, fold_auc = R.oof_binary(df, feats, inj, True, 42)
        A[f'{mod}|{inj}'] = (roc_auc_score(y, p), mw_auc(y, p), PRIM['injuries'][inj][f'{mod}_auc'])
        C[f'{mod}|{inj}'] = (float(np.mean(fold_auc)), float(np.std(fold_auc, ddof=1)), roc_auc_score(y, p))
        d = pd.DataFrame({'s': df.sub_id, 'y': y, 'p': p}).groupby('s').agg(y=('y', 'first'), p=('p', 'mean'))
        E[f'{mod}|{inj}'] = (roc_auc_score(d.y, d.p), len(d))
print('A. AUC: sklearn vs Mann-Whitney vs reported (max abs diff):',
      max(max(abs(a - b), abs(a - c)) for a, b, c in A.values()))
# B
try:
    from statsmodels.stats.multitest import multipletests
    pv = [PRIM['injuries'][i]['p'] for i in R.INJ]
    sm = multipletests(pv, method='holm')[1]
    print('B. Holm reported vs statsmodels:', [round(PRIM['injuries'][i]['p_holm'], 3) for i in R.INJ], [round(x, 3) for x in sm])
except ImportError:
    print('B. statsmodels missing')
print('C. fold-mean AUC (SD) vs pooled:'); [print(f'   {k:14s} {m:.3f} ({s:.3f}) vs {p:.3f}') for k, (m, s, p) in C.items()]
print('E. participant-level AUC (mean prediction per participant):'); [print(f'   {k:14s} {a:.3f} (n={n})') for k, (a, n) in E.items()]

# D other models
def oof_model(df, inj, kind):
    X, y, g = df[feats].values.astype(float), (df['Group'] == inj).astype(int).values, df['sub_id'].values
    p = np.zeros(len(y))
    for tr, te in R.splitter(True, 42).split(X, y, g):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        if kind == 'lr':
            m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000)
        else:
            m = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight='balanced_subsample', random_state=42, n_jobs=1)
        m.fit(Xtr, y[tr]); p[te] = m.predict_proba(Xte)[:, 1]
    return roc_auc_score(y, p)
D = {}
jobs = [(kind, mod, inj) for kind in ('lr', 'rf') for mod, _ in MODS for inj in R.INJ]
vals = Parallel(n_jobs=-1)(delayed(oof_model)(dict(MODS)[mod], inj, kind) for kind, mod, inj in jobs)
for (kind, mod, inj), v in zip(jobs, vals): D[f'{kind}|{mod}|{inj}'] = v
print('D. other models under participant-grouped CV (AUC): XGB / LR / RF')
for mod, _ in MODS:
    for inj in R.INJ:
        print(f'   {mod:4s} {inj:9s} {PRIM["injuries"][inj][f"{mod}_auc"]:.3f} / {D[f"lr|{mod}|{inj}"]:.3f} / {D[f"rf|{mod}|{inj}"]:.3f}')

json.dump(dict(A=A, C=C, D=D, E=E), open(os.path.join(OUT, 'audit.json'), 'w'), indent=1, default=float)
