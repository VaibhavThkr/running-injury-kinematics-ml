"""Reproduce every result, table and figure of the paper from the feature tables in data/.

usage:
    python run_all.py            # full run (1000-permutation test; about 1 h on 2 CPU cores, faster with more)
    python run_all.py --quick    # smoke test with 50 permutations (about 15 min); permutation p values will differ

Outputs go to results/. At the end the main numbers are compared with expected/ (the values reported in the paper).
"""
import os, sys, json, subprocess, time, shutil
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA, RES, SRC, EXP = (os.path.join(ROOT, d) for d in ('data', 'results', 'src', 'expected'))
FIG = os.path.join(RES, 'figures')
QUICK = '--quick' in sys.argv
N_PERM = 50 if QUICK else 1000
os.makedirs(FIG, exist_ok=True)

STEPS = [
    ('Primary and sensitivity analyses (Table 2 numbers, Supplementary Table S1)', ['reanalysis.py', DATA, RES]),
    ('Tables 2, S1a, S1b and Figs 5-8 and 10', ['make_results.py', DATA, RES]),
    (f'Label-permutation test against chance ({N_PERM} permutations)', ['perm_test.py', DATA, RES, str(N_PERM)]),
    ('Constant variables (36 of 76) and identical predictions with the 40 others', ['feature_checks.py', DATA, RES]),
    ('Participant-grouped folds: leakage check', ['leakage_check.py', DATA, RES]),
    ('Methodological checks (AUC formula, Holm, fold-mean AUC, participant-level AUC, other models)', ['audit.py', DATA, RES]),
    ('Values of the originally submitted analysis (trial-level CV)', ['reproduce_original.py', DATA]),
    ('Supplementary Table S2 (variables, units, mean +- SD)', ['variables.py', DATA, RES]),
    ('Table 1 (except the sex distribution) and values given only in the text', ['text_values.py', DATA, RES]),
    ('Fig. 1', ['make_fig1.py', FIG]), ('Fig. 2', ['make_fig2.py', FIG]), ('Fig. 3', ['make_fig3.py', DATA, FIG]), ('Fig. 4', ['make_fig4.py', FIG]),
    ('Fig. 9', ['make_fig9.py', DATA, FIG]),
]

t_all = time.time()
for title, (script, *args) in STEPS:
    t = time.time(); print(f'\n=== {title}', flush=True)
    log = os.path.join(RES, f'log_{os.path.splitext(script)[0]}.txt')
    with open(log, 'w') as fh:
        r = subprocess.run([sys.executable, os.path.join(SRC, script), *args], stdout=fh, stderr=subprocess.STDOUT, cwd=RES)
    if r.returncode:
        print(open(log).read()[-3000:]); sys.exit(f'{script} failed (see {log})')
    print(f'    done in {time.time() - t:.0f} s (log: results/{os.path.basename(log)})', flush=True)
perm = os.path.join(RES, f'permutation_{N_PERM}.json')

# ---------------- compare with the values reported in the paper ----------------
print('\n=== Comparison with expected/ (values reported in the paper)')
ok = True
def flat(x, pre=''):
    if isinstance(x, dict):
        for k, v in x.items(): yield from flat(v, f'{pre}/{k}')
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from flat(v, f'{pre}[{i}]')
    elif isinstance(x, (int, float)) and not isinstance(x, bool): yield pre, float(x)
new = {r['config']: r for r in json.load(open(os.path.join(RES, 'reanalysis_results.json')))['results']}
exp = {r['config']: r for r in json.load(open(os.path.join(EXP, 'reanalysis_results.json')))['results']}
d = max(abs(v - dict(flat(new[c]))[k]) for c in exp for k, v in flat(exp[c]))
print(f'reanalysis_results.json: max |difference| over all configurations = {d:.2e}'); ok &= d < 1e-9
for f in ('Table2_corrected.csv', 'TableS1a_cv_design_and_sample.csv', 'TableS1b_feature_sets.csv', 'TableS2_variables.csv',
          'Table1_from_feature_tables.csv'):
    same = open(os.path.join(RES, f)).read() == open(os.path.join(EXP, f)).read(); ok &= same
    print(f'{f}: {"identical" if same else "DIFFERENT"}')
tv, tve = json.load(open(os.path.join(RES, 'text_values.json'))), json.load(open(os.path.join(EXP, 'text_values.json')))
dtv = max(abs(v - dict(flat(tv))[k]) for k, v in flat(tve)); same_keys = set(tv) == set(tve); ok &= dtv < 1e-9 and same_keys
print(f'text_values.json: max |difference| = {dtv:.2e}{"" if same_keys else " (keys differ)"}')
P, PE = json.load(open(perm)), json.load(open(os.path.join(EXP, 'permutation_1000_summary.json')))
if QUICK:
    print('permutation test: quick mode, p values not compared (the paper uses 1000 permutations)')
else:
    dp = max(abs(P[k]['p'] - PE[k]['p']) for k in PE); ok &= dp < 1e-12
    print(f'permutation p values: max |difference| = {dp:.2e}')
print(f'\n{"ALL CHECKS PASSED" if ok else "SOME OUTPUTS DIFFER - see above"}  (total {time.time() - t_all:.0f} s)')
sys.exit(0 if ok else 1)
