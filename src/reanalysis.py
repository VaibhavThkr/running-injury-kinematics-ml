"""Re-analysis for the revision: participant-grouped cross-validation, participant-level (cluster) bootstrap,
paired walking-vs-running comparison with Holm correction, and multiclass metrics with CIs.

usage: python reanalysis.py <data_dir> <out_dir>

Everything except the fold assignment matches the original pipeline (02_statistical_analysis.py /
05_generate_roc_and_cm.py): same rows, same features, StandardScaler fitted on the training fold, XGBoost with fixed
hyperparameters (100 trees, max depth 3, learning rate 0.1, random_state 42). Changes:
  * folds: StratifiedGroupKFold(5, shuffle=True, random_state=42) grouped by participant (sub_id)
  * scale_pos_weight computed from the training fold only
  * uncertainty: participant-level bootstrap (2000 resamples) of pooled out-of-fold predictions
  * walking-vs-running: paired participant-level bootstrap over all 393 participants; Holm across 5 injuries
Sensitivity configurations: A trial-level CV; C/D without speed-related variables; E 323 participants with both
modalities; F every variable residualized on treadmill speed within each training fold; G/H treadmill speed (and age,
mass and height) added as predictors; I age, mass and height only.
"""
import sys, json, platform, numpy as np, pandas as pd, xgboost as xgb, sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

DATA, OUT = sys.argv[1], sys.argv[2]
SEED, N_BOOT = 42, 2000
INJ = ['ITBS', 'PFPS', 'Plantar', 'Achilles', 'MTSS']          # Group labels in the data
CLASS_ORDER = ['PFPS', 'ITBS', 'Achilles', 'Plantar', 'MTSS']   # Target 0..4
NON_FEATURES = ['Target', 'sub_id', 'Group', 'Age', 'Mass', 'Height', 'Sex', 'Speed']
SPEED_PROXIES = ['STRIDE_LENGTH', 'STRIDE_RATE']                # speed = length x rate / 60 exactly
SPATIOTEMPORAL = ['STEP_WIDTH', 'STRIDE_RATE', 'STRIDE_LENGTH', 'SWING_TIME', 'STANCE_TIME', 'VERTICAL_OSCILLATION']
DEMOGRAPHICS = ['Age', 'Mass', 'Height']   # 'Sex' is 0 in every row of the feature tables, so it is not used

def load(mod):
    return pd.read_csv(f'{DATA}/ml_{mod}_dataset.csv').dropna().reset_index(drop=True)

def features(df, drop=(), add=(), only=None):
    """kinematic feature list (76 columns) minus `drop`, plus `add`; `only` gives an explicit list instead"""
    if only is not None:
        return list(only)
    return [c for c in df.columns if c not in NON_FEATURES and c not in drop] + list(add)

def residualize_on_speed(df, Xtr, Xte, tr, te):
    """remove the linear dependence of every column on treadmill speed; regression fitted on the training fold only"""
    from sklearn.linear_model import LinearRegression
    s = df['Speed'].values.reshape(-1, 1)
    lr = LinearRegression().fit(s[tr], Xtr)
    return Xtr - lr.predict(s[tr]), Xte - lr.predict(s[te])

def splitter(grouped, seed):
    return (StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed) if grouped
            else StratifiedKFold(n_splits=5, shuffle=True, random_state=seed))

def oof_binary(df, feats, inj, grouped=True, seed=SEED, residualize=False):
    X, y, g = df[feats], (df['Group'] == inj).astype(int).values, df['sub_id'].values
    p, fold_auc = np.zeros(len(y)), []
    for tr, te in splitter(grouped, seed).split(X, y, g if grouped else None):
        if residualize:
            Xtr0, Xte0 = residualize_on_speed(df, X.values[tr].astype(float), X.values[te].astype(float), tr, te)
            sc = StandardScaler(); Xtr = sc.fit_transform(Xtr0); Xte = sc.transform(Xte0)
        else:
            sc = StandardScaler(); Xtr = sc.fit_transform(X.iloc[tr]); Xte = sc.transform(X.iloc[te])
        spw = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())
        clf = xgb.XGBClassifier(eval_metric='logloss', max_depth=3, n_estimators=100, learning_rate=0.1,
                                scale_pos_weight=spw, random_state=42, n_jobs=1)
        clf.fit(Xtr, y[tr]); p[te] = clf.predict_proba(Xte)[:, 1]
        fold_auc.append(roc_auc_score(y[te], p[te]))
    return y, p, fold_auc

def oof_multiclass(df, feats, grouped=True, seed=SEED, residualize=False):
    X, y, g = df[feats], df['Target'].values.astype(int), df['sub_id'].values
    pred, fold_acc = np.zeros(len(y), dtype=int), []
    for tr, te in splitter(grouped, seed).split(X, y, g if grouped else None):
        if residualize:
            Xtr0, Xte0 = residualize_on_speed(df, X.values[tr].astype(float), X.values[te].astype(float), tr, te)
            sc = StandardScaler(); Xtr = sc.fit_transform(Xtr0); Xte = sc.transform(Xte0)
        else:
            sc = StandardScaler(); Xtr = sc.fit_transform(X.iloc[tr]); Xte = sc.transform(X.iloc[te])
        clf = xgb.XGBClassifier(eval_metric='mlogloss', objective='multi:softprob', num_class=5, max_depth=3,
                                n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=1)
        clf.fit(Xtr, y[tr]); pred[te] = clf.predict(Xte)
        fold_acc.append(accuracy_score(y[te], pred[te]))
    return y, pred, fold_acc

# ---------- participant-level bootstrap machinery ----------
def participant_index(df):
    return {s: np.flatnonzero(df['sub_id'].values == s) for s in df['sub_id'].unique()}

def draw_indices(pidx, drawn):
    parts = [pidx[s] for s in drawn if s in pidx]
    return np.concatenate(parts) if parts else np.array([], dtype=int)

def boot_p(d):
    """two-sided bootstrap p for H0: difference = 0 (add-one, capped at 1)"""
    B = len(d)
    return min(1.0, 2 * min((1 + np.sum(d <= 0)) / (B + 1), (1 + np.sum(d >= 0)) / (B + 1)))

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p); adj = np.empty(m); running = 0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i])); adj[i] = running
    return adj

def run_config(name, drop=(), grouped=True, seed=SEED, n_boot=N_BOOT, subset_both=False, add=(), only=None, residualize=False):
    run, walk = load('run'), load('walk')
    fr, fw = features(run, drop, add, only), features(walk, drop, add, only)
    assert fr == fw
    if 'Height' in fr:   # 999 is a missing-value code in the source metadata (2 participants); scaler and XGBoost handle NaN
        for df in (run, walk): df.loc[df['Height'] >= 300, 'Height'] = np.nan
    res = {'config': name, 'n_features': len(fr), 'grouped': grouped, 'residualized_on_speed': residualize, 'injuries': {}}
    preds = {}
    for inj in INJ:
        yr, pr, far = oof_binary(run, fr, inj, grouped, seed, residualize)
        yw, pw, faw = oof_binary(walk, fw, inj, grouped, seed, residualize)
        preds[inj] = (yr, pr, yw, pw)
        res['injuries'][inj] = dict(run_auc=roc_auc_score(yr, pr), walk_auc=roc_auc_score(yw, pw),
                                    run_fold_sd=float(np.std(far, ddof=1)), walk_fold_sd=float(np.std(faw, ddof=1)),
                                    run_ap=average_precision_score(yr, pr), walk_ap=average_precision_score(yw, pw),
                                    run_prev=float(yr.mean()), walk_prev=float(yw.mean()))
    # paired participant-level bootstrap over the union of participants
    pid_r, pid_w = participant_index(run), participant_index(walk)
    universe = np.array(sorted(set(run.sub_id) | set(walk.sub_id)))
    if subset_both:
        universe = np.array(sorted(set(run.sub_id) & set(walk.sub_id)))
    rng = np.random.default_rng(seed)
    boots = {inj: {'run': [], 'walk': [], 'diff': []} for inj in INJ}
    for _ in range(n_boot):
        drawn = rng.choice(universe, size=len(universe), replace=True)
        ir, iw = draw_indices(pid_r, drawn), draw_indices(pid_w, drawn)
        for inj in INJ:
            yr, pr, yw, pw = preds[inj]
            try:
                ar, aw = roc_auc_score(yr[ir], pr[ir]), roc_auc_score(yw[iw], pw[iw])
            except ValueError:
                continue
            boots[inj]['run'].append(ar); boots[inj]['walk'].append(aw); boots[inj]['diff'].append(aw - ar)
    pv = []
    for inj in INJ:
        b = {k: np.array(v) for k, v in boots[inj].items()}
        r = res['injuries'][inj]
        if subset_both:   # point estimates on the paired subset
            keep_r = np.isin(run.sub_id, universe); keep_w = np.isin(walk.sub_id, universe)
            yr, pr, yw, pw = preds[inj]
            r['run_auc'] = roc_auc_score(yr[keep_r], pr[keep_r]); r['walk_auc'] = roc_auc_score(yw[keep_w], pw[keep_w])
        r['run_ci'] = list(np.percentile(b['run'], [2.5, 97.5])); r['walk_ci'] = list(np.percentile(b['walk'], [2.5, 97.5]))
        r['diff'] = r['walk_auc'] - r['run_auc']; r['diff_ci'] = list(np.percentile(b['diff'], [2.5, 97.5]))
        r['p'] = boot_p(b['diff']); pv.append(r['p'])
    for inj, a in zip(INJ, holm(pv)):
        res['injuries'][inj]['p_holm'] = float(a)
    # multiclass
    mc = {}
    for mod, df in [('run', run), ('walk', walk)]:
        y, pred, facc = oof_multiclass(df, fr, grouped, seed, residualize)
        cm = confusion_matrix(y, pred, labels=range(5))
        pidx = participant_index(df); uni = np.array(sorted(df.sub_id.unique())); rng2 = np.random.default_rng(seed)
        acc_b, bal_b, f1_b = [], [], []
        for _ in range(n_boot):
            idx = draw_indices(pidx, rng2.choice(uni, size=len(uni), replace=True))
            acc_b.append(accuracy_score(y[idx], pred[idx])); bal_b.append(balanced_accuracy_score(y[idx], pred[idx]))
            f1_b.append(f1_score(y[idx], pred[idx], average='macro'))
        counts = np.bincount(y, minlength=5)
        mc[mod] = dict(acc=accuracy_score(y, pred), acc_ci=list(np.percentile(acc_b, [2.5, 97.5])), acc_fold_sd=float(np.std(facc, ddof=1)),
                       bal_acc=balanced_accuracy_score(y, pred), bal_ci=list(np.percentile(bal_b, [2.5, 97.5])),
                       macro_f1=f1_score(y, pred, average='macro'), f1_ci=list(np.percentile(f1_b, [2.5, 97.5])),
                       majority=float(counts.max() / counts.sum()), majority_class=CLASS_ORDER[int(counts.argmax())],
                       recall={CLASS_ORDER[k]: float(cm[k, k] / cm[k].sum()) for k in range(5)},
                       cm=cm.tolist())
    res['multiclass'] = mc
    return res

def fmt(res):
    lines = [f"\n=== {res['config']}  ({res['n_features']} features, {'participant-grouped' if res['grouped'] else 'flat trial-level'} CV) ==="]
    lines.append(f"{'Injury':9s} {'Walk AUC [95% CI]':>22s} {'Run AUC [95% CI]':>22s} {'Walk-Run [95% CI]':>24s} {'p':>6s} {'Holm':>6s}")
    for inj in INJ:
        r = res['injuries'][inj]
        lines.append(f"{inj:9s} {r['walk_auc']:.3f} [{r['walk_ci'][0]:.3f}, {r['walk_ci'][1]:.3f}] "
                     f"{r['run_auc']:.3f} [{r['run_ci'][0]:.3f}, {r['run_ci'][1]:.3f}] "
                     f"{r['diff']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}] {r['p']:6.3f} {r['p_holm']:6.3f}")
    for mod in ['run', 'walk']:
        m = res['multiclass'][mod]
        lines.append(f"multiclass {mod}: acc {m['acc']:.3f} [{m['acc_ci'][0]:.3f}, {m['acc_ci'][1]:.3f}] (fold SD {m['acc_fold_sd']:.3f}); "
                     f"balanced {m['bal_acc']:.3f} [{m['bal_ci'][0]:.3f}, {m['bal_ci'][1]:.3f}]; macro-F1 {m['macro_f1']:.3f}; "
                     f"majority ({m['majority_class']}) {m['majority']:.3f}; recall " +
                     ", ".join(f"{k} {v:.3f}" for k, v in m['recall'].items()))
    return "\n".join(lines)

if __name__ == '__main__':
    import os; os.makedirs(OUT, exist_ok=True)
    configs = [
        dict(name='A_original_flat_76', grouped=False),
        dict(name='B_grouped_76', grouped=True),
        dict(name='C_grouped_74_no_stride_length_rate', grouped=True, drop=SPEED_PROXIES),
        dict(name='D_grouped_70_no_spatiotemporal', grouped=True, drop=SPATIOTEMPORAL),
        dict(name='E_grouped_76_paired_subset_323', grouped=True, subset_both=True),
        # speed and demographics (Reviewer 1, 'Kinematic variables')
        dict(name='F_grouped_76_speed_residualized', grouped=True, residualize=True),
        dict(name='G_grouped_77_plus_speed', grouped=True, add=['Speed']),
        dict(name='H_grouped_80_plus_age_mass_height_speed', grouped=True, add=DEMOGRAPHICS + ['Speed']),
        dict(name='I_grouped_age_mass_height_only', grouped=True, only=DEMOGRAPHICS),
    ]
    allres = []
    for c in configs:
        r = run_config(c['name'], drop=c.get('drop', ()), grouped=c['grouped'], subset_both=c.get('subset_both', False),
                       add=c.get('add', ()), only=c.get('only'), residualize=c.get('residualize', False))
        print(fmt(r), flush=True); allres.append(r)
    meta = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                xgboost=xgb.__version__, seed=SEED, n_boot=N_BOOT)
    json.dump(dict(meta=meta, results=allres), open(f'{OUT}/reanalysis_results.json', 'w'), indent=1, default=float)
    print('\n', meta)
