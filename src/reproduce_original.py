"""Reproduce the ORIGINAL submission's Table 2 and Fig 5 numbers exactly (flat StratifiedKFold on trial rows),
following 02_statistical_analysis.py and 05_generate_roc_and_cm.py line by line. Usage: python reproduce_original.py <data_dir>"""
import sys, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DATA = sys.argv[1]
EXCLUDE_COLS = ['Target', 'sub_id', 'Group', 'Age', 'Mass', 'Height', 'Sex', 'Speed']

def get_cv_probs(df, target_inj):
    X = df.drop(EXCLUDE_COLS, axis=1, errors='ignore')
    y_bin = (df['Group'] == target_inj).astype(int)
    pos_weight = (len(y_bin) - sum(y_bin)) / sum(y_bin) if sum(y_bin) > 0 else 1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = np.zeros(len(y_bin))
    for train, test in cv.split(X, y_bin):
        sc = StandardScaler(); Xtr = sc.fit_transform(X.iloc[train]); Xte = sc.transform(X.iloc[test])
        clf = xgb.XGBClassifier(eval_metric='logloss', max_depth=3, n_estimators=100, learning_rate=0.1,
                                scale_pos_weight=pos_weight, random_state=42)
        clf.fit(Xtr, y_bin.iloc[train]); probs[test] = clf.predict_proba(Xte)[:, 1]
    return y_bin.values, probs

def bootstrap_auc(y, p, n_boot=1000, seed=42):
    np.random.seed(seed); aucs = []
    for _ in range(n_boot):
        idx = np.random.choice(len(y), len(y), replace=True)
        try: aucs.append(roc_auc_score(y[idx], p[idx]))
        except ValueError: pass
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def bootstrap_auc_diff(y_run, p_run, y_walk, p_walk, n_boot=1000, seed=42):
    np.random.seed(seed); diffs = []
    for _ in range(n_boot):
        ir = np.random.choice(len(y_run), len(y_run), replace=True)
        iw = np.random.choice(len(y_walk), len(y_walk), replace=True)
        try: diffs.append(roc_auc_score(y_walk[iw], p_walk[iw]) - roc_auc_score(y_run[ir], p_run[ir]))
        except ValueError: pass
    d = np.array(diffs); p = 2 * min(np.mean(d > 0), np.mean(d < 0))
    return np.mean(d), np.percentile(d, 2.5), np.percentile(d, 97.5), p

run_df = pd.read_csv(f'{DATA}/ml_run_dataset.csv').dropna()
walk_df = pd.read_csv(f'{DATA}/ml_walk_dataset.csv').dropna()
print(f"{'Injury':9s} {'Walk mean±hw':>15s} {'Run mean±hw':>15s} {'Diff [95% CI]':>26s} {'p':>6s} | pooled AUC walk / run")
for inj in ['ITBS', 'PFPS', 'Plantar', 'Achilles', 'MTSS']:
    yr, pr = get_cv_probs(run_df, inj); yw, pw = get_cv_probs(walk_df, inj)
    rm, rl, ru = bootstrap_auc(yr, pr); wm, wl, wu = bootstrap_auc(yw, pw)
    dm, dl, du, p = bootstrap_auc_diff(yr, pr, yw, pw)
    print(f"{inj:9s} {wm:.3f} ± {(wu-wl)/2:.3f}  {rm:.3f} ± {(ru-rl)/2:.3f}  {dm:+.3f} [{dl:+.3f}, {du:+.3f}] {p:6.3f} | "
          f"{roc_auc_score(yw, pw):.3f} / {roc_auc_score(yr, pr):.3f}")
