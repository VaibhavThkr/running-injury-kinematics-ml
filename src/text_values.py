"""Values reported in the text that are not part of a table or figure, and Table 1 (except the sex distribution).
usage: python text_values.py <data_dir> <out_dir>   (after reanalysis.py)

  * trials per participant, repeat sessions and participants per injury group in each modality (Sections 2.1, 3.4, 4.1)
  * Spearman correlation between the trial-level minus participant-grouped AUC and the share of trials that are repeat
    sessions, across the 10 injury-modality combinations (Section 3.4)
  * the range covered by 90% of trials (5th-95th percentile) of treadmill speed in each modality (Section 3.3)
  * the 2 walking trials with spatiotemporal values typical of running, and the walking AUCs without them, with the fold
    assignment of the primary analysis unchanged (Section 4.1)
  * Table 1: age, mass, height, BMI and speed by injury group. Height is missing (code 999) for 2 participants, whose
    height and BMI are excluded. Sex is not coded in the feature tables, so the 'Male (%)' column is not reproduced here.
"""
import os, sys, json, numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reanalysis as R

DATA, OUT = sys.argv[1], sys.argv[2]
CFG = {r['config']: r for r in json.load(open(os.path.join(OUT, 'reanalysis_results.json')))['results']}
PRIM, FLAT = CFG['B_grouped_76'], CFG['A_original_flat_76']
ORDER = ['PFPS', 'ITBS', 'Plantar', 'Achilles', 'MTSS']
run, walk = R.load('run'), R.load('walk')
res = {}

# ---- participants, trials and repeat sessions ----
for mod, df in (('run', run), ('walk', walk)):
    per = df.groupby('sub_id').size()
    res[f'{mod}_trials_per_participant'] = {int(k): int(v) for k, v in per.value_counts().sort_index().items()}
    res[f'{mod}_participants_with_repeats'] = int((per > 1).sum())
    res[f'{mod}_trials_from_participants_with_repeats'] = int(per[per > 1].sum())
    for inj in ORDER:
        g = df[df.Group == inj]
        res[f'{mod}_{inj}_trials_participants'] = [len(g), int(g.sub_id.nunique())]
both = set(run.sub_id) & set(walk.sub_id)
res['participants_both_run_only_walk_only'] = [len(both), len(set(run.sub_id) - both), len(set(walk.sub_id) - both)]

# ---- inflation vs share of trials that are repeat sessions ----
share, infl = [], []
for mod, df in (('run', run), ('walk', walk)):
    for inj in ORDER:
        g = df[df.Group == inj]
        share.append((len(g) - g.sub_id.nunique()) / len(g))
        infl.append(FLAT['injuries'][inj][f'{mod}_auc'] - PRIM['injuries'][inj][f'{mod}_auc'])
rho, p = spearmanr(share, infl)
res['spearman_inflation_vs_repeat_share'] = dict(rho=float(rho), p=float(p))

# ---- speed ----
for mod, df in (('run', run), ('walk', walk)):
    res[f'{mod}_speed_p5_p95'] = [float(x) for x in np.percentile(df.Speed, [5, 95])]

# ---- 2 walking trials with running-like spatiotemporal values ----
odd = np.flatnonzero(walk.Speed.values > 2.0)
res['walking_trials_over_2ms'] = walk.loc[odd, ['sub_id', 'Group', 'Speed', 'STANCE_TIME']].round(3).values.tolist()
feats = R.features(walk); drop = set(odd.tolist()); res['walk_auc_without_them_same_folds'] = {}
for inj in R.INJ:
    X, y, g = walk[feats], (walk['Group'] == inj).astype(int).values, walk['sub_id'].values
    p_ = np.full(len(y), np.nan)
    for tr, te in R.splitter(True, R.SEED).split(X, y, g):
        tr = np.array([i for i in tr if i not in drop]); te = np.array([i for i in te if i not in drop])
        sc = StandardScaler(); Xtr = sc.fit_transform(X.iloc[tr]); Xte = sc.transform(X.iloc[te])
        spw = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())
        clf = xgb.XGBClassifier(eval_metric='logloss', max_depth=3, n_estimators=100, learning_rate=0.1,
                                scale_pos_weight=spw, random_state=42, n_jobs=1)
        clf.fit(Xtr, y[tr]); p_[te] = clf.predict_proba(Xte)[:, 1]
    k = ~np.isnan(p_); a = roc_auc_score(y[k], p_[k])
    res['walk_auc_without_them_same_folds'][inj] = [float(a), float(a - PRIM['injuries'][inj]['walk_auc'])]

# ---- Table 1 (participant level; first recorded session where demographics differ between sessions) ----
P = pd.concat([run, walk]).groupby('sub_id').first()
P['H'] = P.Height.where(P.Height < 300) / 100; P['BMI'] = P.Mass / P.H ** 2
rows = []
for inj in ORDER:
    x = P[P.Group == inj]
    f = lambda s, d: f'{s.mean():.{d}f} ± {s.std():.{d}f}'
    rows.append(dict(Group=inj, n=len(x), Age=f(x.Age, 1), Mass=f(x.Mass, 1), Height=f(x.H, 2), BMI=f(x.BMI, 1),
                     Height_missing=int(x.H.isna().sum()),
                     Walk_speed=f(walk[walk.Group == inj].Speed, 2), Run_speed=f(run[run.Group == inj].Speed, 2)))
T1 = pd.DataFrame(rows); T1.to_csv(os.path.join(OUT, 'Table1_from_feature_tables.csv'), index=False)
print(T1.to_string(index=False))
json.dump(res, open(os.path.join(OUT, 'text_values.json'), 'w'), indent=1)
for k, v in res.items(): print(k, v)
