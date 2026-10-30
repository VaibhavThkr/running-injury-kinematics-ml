"""Constant variables: list the variables that are 0 in every walking and running trial, and confirm that removing them
leaves every model prediction unchanged (trees cannot split on a constant column).
usage: python feature_checks.py <data_dir> <out_dir>   -> <out_dir>/feature_checks.json"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reanalysis as R

run, walk = R.load('run'), R.load('walk')
F76 = R.features(run)
CONST = [c for c in F76 if run[c].nunique() == 1 and walk[c].nunique() == 1]
F40 = [c for c in F76 if c not in CONST]
assert all(run[c].iloc[0] == 0 and walk[c].iloc[0] == 0 for c in CONST)
max_diff = 0.0
for df in (run, walk):
    for inj in R.INJ:
        _, p76, _ = R.oof_binary(df, F76, inj); _, p40, _ = R.oof_binary(df, F40, inj)
        max_diff = max(max_diff, float(np.max(np.abs(p76 - p40))))
    _, c76, _ = R.oof_multiclass(df, F76); _, c40, _ = R.oof_multiclass(df, F40)
    assert (c76 == c40).all()
print(f'{len(CONST)} of {len(F76)} variables are 0 in every trial; {len(F40)} vary.')
print(f'max |difference| in one-versus-rest predicted probability, 76 vs 40 variables: {max_diff:.2e}; multi-class predictions identical')
json.dump(dict(n_constant=len(CONST), constant=CONST, informative=F40, max_abs_prob_diff_76_vs_40=max_diff),
          open(os.path.join(sys.argv[2], 'feature_checks.json'), 'w'), indent=1)
