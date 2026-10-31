# Walking versus running kinematics for classifying running-related injuries

## What the analysis does

* **Data:** 393 participants from the Running Injury Clinic Kinematic Dataset (Ferber et al., *Sci Data* 2024;11:1232)
  with a primary diagnosis of patellofemoral pain syndrome (PFPS), iliotibial band syndrome (ITBS), plantar fasciitis,
  Achilles tendinopathy or medial tibial stress syndrome (MTSS): 434 running trials (366 participants) and 391 walking
  trials (350 participants).
* **Predictors:** the 76 discrete variables provided by the source dataset for each session (`data/variables.csv`).
  36 of them are 0 in every trial and cannot influence any model; age, mass, height, sex and treadmill speed are not
  predictors in the primary analysis.
* **Models:** XGBoost with fixed hyperparameters (100 trees, maximum depth 3, learning rate 0.1): 5 one-versus-rest
  models and a 5-way multi-class model, fitted separately to walking and running trials.
* **Validation:** participant-grouped, stratified 5-fold cross-validation (`StratifiedGroupKFold`, seed 42), so that all
  sessions of a participant are in the same fold. Feature scaling and class weights are estimated in the training folds.
* **Inference:** participant-level bootstrap (2000 resamples) for 95% CIs; paired participant-level bootstrap for the
  walking-minus-running AUC difference with Holm correction across the 5 injuries; a label-permutation test against
  chance (1000 permutations of the injury labels across participants, Holm-adjusted within each modality).
* **Sensitivity analyses:** conventional trial-level cross-validation (which lets repeat sessions of a participant fall
  in both training and test folds), 20 fold assignments, 1 session per participant, the 323 participants with both
  modalities, removal of speed-related variables, speed-residualized variables, and treadmill speed, age, body mass and
  height added as predictors (or age, body mass and height used alone).

## Repository layout

| Path | Contents |
|---|---|
| `data/ml_run_dataset.csv`, `data/ml_walk_dataset.csv` | one row per session: participant ID (`sub_id`), diagnosis (`Group`, `Target`), demographics, treadmill speed and the 76 variables |
| `data/variables.csv` | definition, plane, unit and mean ± SD of each variable (Supplementary Table S2) |
| `feature_extraction/` | code that builds the two feature tables from the public JSON files |
| `src/reanalysis.py` | primary analysis and all sensitivity configurations (results/reanalysis_results.json) |
| `src/make_results.py` | Table 2, Supplementary Tables S1a and S1b, Figs 5–8 and 10 |
| `src/make_fig9.py` | Fig. 9 (treadmill speed by modality and injury group) |
| `src/text_values.py` | Table 1 (except the sex distribution) and the values given only in the text: trials per participant, repeat sessions, the correlation of the cross-validation inflation with the share of repeat sessions, the speed ranges, and the walking AUCs without the 2 walking trials with running-like values |
| `src/perm_test.py` | label-permutation test of each one-versus-rest AUC against chance |
| `src/feature_checks.py` | the 36 constant variables; identical predictions with the remaining 40 |
| `src/leakage_check.py` | confirms that no participant appears in both training and test data of any grouped fold |
| `src/audit.py` | AUC by the Mann–Whitney formula, Holm against statsmodels, fold-mean vs pooled AUC, participant-level AUC, logistic regression and random forest |
| `src/reproduce_original.py` | reproduces the values of the originally submitted analysis (trial-level cross-validation) |
| `src/variables.py`, `src/make_fig1.py` … `make_fig4.py` | Supplementary Table S2 and Figs 1–4 |
| `expected/` | the values reported in the paper, used by `run_all.py` to check the reproduction |

## How to run

Python 3.13 was used; install the pinned packages and run everything:

```bash
python -m pip install -r requirements.txt
python run_all.py            # full run, about 1 h on 2 CPU cores (most of it the 1000-permutation test)
python run_all.py --quick    # 50 permutations, about 15 min
```

All outputs are written to `results/` (tables as CSV, figures as PNG, TIFF and PDF, logs). The run ends by comparing the
results with `expected/` and prints `ALL CHECKS PASSED` when every value matches. Individual scripts can also be run on
their own, for example `python src/reanalysis.py data results`.

Results are deterministic: fold assignment uses seed 42, XGBoost uses `random_state=42` and `n_jobs=1`, the bootstrap
uses `numpy.random.default_rng(42)`, and permutation *i* uses seed *i*. Figures use the Liberation Sans font and fall back
to the matplotlib default if it is not installed.

## Data source and license

The feature tables are derived from the Running Injury Clinic Kinematic Dataset, Figshare+,
https://doi.org/10.25452/figshare.plus.24255795.v1, released under CC BY 4.0, and described in Ferber R, Brett A,
Fukuchi RK, Hettinga B, Osis ST. A biomechanical dataset of 1,798 healthy and injured subjects during treadmill walking
and running. *Sci Data* 2024;11:1232. https://doi.org/10.1038/s41597-024-04011-7. Please cite the source dataset when
using these tables. The derived tables are shared under the same license (CC BY 4.0); the code is released under the
MIT license (`LICENSE`).
