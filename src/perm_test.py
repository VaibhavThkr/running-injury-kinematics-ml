"""Label-permutation test for one-versus-rest AUC > 0.5 under participant-grouped CV, with Holm adjustment.
usage: python perm_test.py <data_dir> <out_dir> <n_perm>
Injury labels are permuted across participants (each participant's trials keep one label); the complete
participant-grouped 5-fold CV (fold seed 42, same model as the primary analysis) is repeated per permutation.
p = (1 + #null >= observed) / (n_perm + 1), one-sided. Holm adjustment over the 5 injuries within each modality,
and over all 10 injury-modality tests for reference. Full null distributions are saved so the test can be extended.
"""
import sys, os, json, time, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA, OUT, NPERM = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.argv = [sys.argv[0], DATA, OUT]
import reanalysis as R

J = json.load(open(os.path.join(OUT, 'reanalysis_results.json')))
PRIM = {r['config']: r for r in J['results']}['B_grouped_76']
MODS = [('run', R.load('run')), ('walk', R.load('walk'))]
feats = R.features(MODS[0][1])


def perm_auc(df, inj, seed):
    rng = np.random.default_rng(seed)
    lab = df.groupby('sub_id')['Group'].first()
    perm = pd.Series(rng.permutation(lab.values), index=lab.index)
    d2 = df.copy(); d2['Group'] = d2['sub_id'].map(perm)
    y, p, _ = R.oof_binary(d2, feats, inj, True, 42)
    return roc_auc_score(y, p)


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run_max = 0.0
    for k, i in enumerate(o):
        run_max = max(run_max, (m - k) * p[i]); adj[i] = min(1.0, run_max)
    return adj


res, t0 = {}, time.time()
for mod, df in MODS:
    for inj in R.INJ:
        null = np.array(Parallel(n_jobs=-1)(delayed(perm_auc)(df, inj, s) for s in range(NPERM)))
        obs = PRIM['injuries'][inj][f'{mod}_auc']
        k = int(np.sum(null >= obs))
        res[f'{mod}|{inj}'] = dict(obs=obs, k=k, n_perm=NPERM, p=(1 + k) / (NPERM + 1),
                                   null_mean=float(null.mean()), null_95=float(np.percentile(null, 95)),
                                   null=[round(float(v), 6) for v in null])
        print(f'{mod:4s} {inj:9s} AUC {obs:.3f}  k={k:4d}  p={(1 + k) / (NPERM + 1):.4f}  '
              f'null mean {null.mean():.3f}, 95th pct {np.percentile(null, 95):.3f}  [{time.time() - t0:.0f} s]', flush=True)

keys = list(res)
for mod, _ in MODS:
    ks = [k for k in keys if k.startswith(mod + '|')]
    for k, a in zip(ks, holm([res[k]['p'] for k in ks])): res[k]['p_holm_within_modality'] = float(a)
for k, a in zip(keys, holm([res[k]['p'] for k in keys])): res[k]['p_holm_all10'] = float(a)
print('\nmodality injury    AUC    p       Holm(5, within modality)  Holm(10)')
for k in keys:
    r = res[k]; mod, inj = k.split('|')
    print(f'{mod:8s} {inj:9s} {r["obs"]:.3f}  {r["p"]:.4f}  {r["p_holm_within_modality"]:.4f}                    {r["p_holm_all10"]:.4f}')
json.dump(res, open(os.path.join(OUT, f'permutation_{NPERM}.json'), 'w'), indent=1)
print('saved', os.path.join(OUT, f'permutation_{NPERM}.json'))
