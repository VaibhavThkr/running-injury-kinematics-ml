# Feature extraction

Place here the script(s) that build `data/ml_run_dataset.csv` and `data/ml_walk_dataset.csv` from the public JSON files
of the Running Injury Clinic Kinematic Dataset (`ric_data.zip`, https://doi.org/10.25452/figshare.plus.24255795.v1).

The extraction should document, as the paper does:

* participant selection (primary diagnosis of 1 of the 5 target injuries; exclusion of healthy controls, non-target or
  unspecified diagnoses, and participants with missing demographic or gait-speed data);
* one row per treadmill session, with the 76 discrete variables taken from the session's descriptive variables;
* limb handling: the injured limb for a unilateral injury, the mean of both limbs for a bilateral injury or an
  unspecified side;
* the 4 running session records excluded for incomplete kinematic data.
