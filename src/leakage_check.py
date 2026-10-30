"""Check that participant-grouped cross-validation never places the same participant in both the training and the test
data of a fold (every one-versus-rest target and the multi-class target, both modalities, fold seed 42 and 20 further seeds),
and count how often trial-level cross-validation does.
usage: python leakage_check.py <data_dir> <out_dir>"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reanalysis as R

out = {}
for mod in ('run', 'walk'):
    df = R.load(mod); feats = R.features(df); g = df['sub_id'].values
    targets = {inj: (df['Group'] == inj).astype(int).values for inj in R.INJ}
    targets['multiclass'] = df['Target'].values.astype(int)
    for seed in [42] + list(range(20)):
        for name, y in targets.items():
            for tr, te in R.splitter(True, seed).split(df[feats], y, g):
                assert not set(g[tr]) & set(g[te]), f'participant overlap in grouped CV ({mod}, {name}, seed {seed})'
    shared_flat = [len(set(g[tr]) & set(g[te])) for tr, te in R.splitter(False, 42).split(df[feats], targets['ITBS'])]
    out[mod] = dict(grouped_overlap=0, flat_participants_in_both_per_fold=shared_flat)
    print(f'{mod}: grouped CV - no participant in both training and test data (21 seeds x 6 targets); '
          f'trial-level CV (ITBS target, seed 42) - participants in both, per fold: {shared_flat}')
json.dump(out, open(os.path.join(sys.argv[2], 'leakage_check.json'), 'w'), indent=1)
