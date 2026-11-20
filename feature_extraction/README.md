# Feature extraction

This directory contains the feature extraction pipeline and architecture diagrams used to construct `data/ml_run_dataset.csv` and `data/ml_walk_dataset.csv` from the public JSON files of the Running Injury Clinic Kinematic Dataset (`ric_data.zip`, https://doi.org/10.25452/figshare.plus.24255795.v1).

## Pipeline Script: `01_extract_dataset.py`

* **Function:** Extracts the 60+ discrete 3D kinematic scalar features (peak flexion/adduction angles, excursions, angular velocities, and initial contact values) from `ric_data.zip` across all 5 overuse injury cohorts (PFPS, ITBS, Achilles Tendinopathy, Plantar Fasciitis, MTSS) and healthy controls.
* **Extraction Methodology:**
  * **Participant selection:** Primary diagnosis of 1 of the 5 target injuries; exclusion of non-target or unspecified diagnoses, and records with missing demographic (`Weight`, `Height`, `age`, `Gender`) or gait speed data.
  * **Discrete kinematic variables:** One row per treadmill session with the discrete variables taken from each session's descriptive variables (`dv_r` for running, `dv_w` for walking).
  * **Limb handling:** Extracts the affected limb for unilateral injuries; computes the bilateral mean of both limbs for bilateral injuries or unspecified sides.
  * **Quality control:** Excludes sessions with incomplete or invalid kinematic data.
* **Output:** Generates structured training datasets (`ml_run_dataset.csv` and `ml_walk_dataset.csv`) saved to `data/`.

## Architecture Figure

* **`Fig04_Feature_Extraction.png`**: High-resolution architectural diagram depicting the end-to-end discrete kinematic feature extraction pipeline, limb assignment logic, and quality filtering stages.
