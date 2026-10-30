# Data

`ml_run_dataset.csv` (434 rows) and `ml_walk_dataset.csv` (391 rows): one row per treadmill session.

| Column | Meaning |
|---|---|
| `sub_id` | participant identifier of the source dataset |
| `Group` | primary diagnosis: `PFPS`, `ITBS`, `Plantar` (plantar fasciitis), `Achilles` (Achilles tendinopathy), `MTSS` |
| `Target` | the same diagnosis as an integer (0 PFPS, 1 ITBS, 2 Achilles, 3 Plantar, 4 MTSS) |
| `Age`, `Mass`, `Height` | age (years), body mass (kg) and height (cm; 999 = missing, 2 participants); not predictors in the primary analysis |
| `Sex` | not coded in these tables (0 in every row) and not used |
| `Speed` | treadmill speed (m/s), equal to stride length × stride rate / 60; not a predictor in the primary analysis |
| 76 further columns | the discrete gait variables of the source dataset; see `variables.csv` |

`variables.csv` gives, for each of the 76 variables, the region, plane, variable type, description, unit and the mean ± SD
in running and walking trials. 36 variables are 0 in every trial (`Constant = True`).

Joint angles are those of the source dataset: Cardan angles of the distal relative to the proximal segment, with
flexion/extension first, abduction/adduction second and internal/external rotation third; the pelvis is expressed
relative to the laboratory (Ferber et al., *Sci Data* 2024;11:1232). Signs follow the source data (for example, knee
flexion is negative).

Source: Running Injury Clinic Kinematic Dataset, https://doi.org/10.25452/figshare.plus.24255795.v1 (CC BY 4.0).
