"""Metadata for the 76 discrete variables in ml_run_dataset.csv / ml_walk_dataset.csv (Supplementary Table S2, Fig. 3).

Names are those of the source dataset (Ferber et al., Sci Data 2024;11:1232). Joint angles are Cardan angles of the distal
relative to the proximal segment (flexion/extension, then abduction/adduction, then internal/external rotation), computed
by the source laboratory's software; signs follow the source data. Units of timing variables were checked against the
value ranges in the data (ANKLE_EVE_percent_STANCE is stored as a fraction of stance, 0-1).
usage: python variables.py <data_dir> <out_dir>   -> writes <out_dir>/TableS2_variables.csv
"""
import os, sys

DEG, DPS, PST = 'deg', 'deg/s', '% stance'
# (name, region, plane, type, description, unit)
VARIABLES = [
    # spatiotemporal
    ('STEP_WIDTH', 'Spatiotemporal', '—', 'Spatiotemporal', 'Step width (mediolateral distance between the feet)', 'm'),
    ('STRIDE_RATE', 'Spatiotemporal', '—', 'Spatiotemporal', 'Stride rate', 'strides/min'),
    ('STRIDE_LENGTH', 'Spatiotemporal', '—', 'Spatiotemporal', 'Stride length', 'm'),
    ('STANCE_TIME', 'Spatiotemporal', '—', 'Spatiotemporal', 'Stance time', 's'),
    ('SWING_TIME', 'Spatiotemporal', '—', 'Spatiotemporal', 'Swing time', 's'),
    ('VERTICAL_OSCILLATION', 'Spatiotemporal', '—', 'Spatiotemporal', 'Vertical oscillation', 'mm'),
    # pelvis
    ('PELVIS_PEAK_DROP_ANGLE', 'Pelvis', 'Frontal', 'Peak angle', 'Peak pelvic drop', DEG),
    ('PELVIS_DROP_EXCURSION', 'Pelvis', 'Frontal', 'Excursion', 'Pelvic drop excursion during stance', DEG),
    ('PELVIS_DROP_at_HS', 'Pelvis', 'Frontal', 'Angle at initial contact', 'Pelvic drop at initial contact', DEG),
    ('PELVIS_DROP_percent_STANCE', 'Pelvis', 'Frontal', 'Timing of peak angle', 'Timing of peak pelvic drop', PST),
    ('PEAK_PELVIC_DROP_VELOCITY', 'Pelvis', 'Frontal', 'Peak angular velocity', 'Peak pelvic drop velocity', DPS),
    # hip
    ('HIP_EXT_PEAK_ANGLE', 'Hip', 'Sagittal', 'Peak angle', 'Peak hip extension', DEG),
    ('HIP_EXT_EXCURSION', 'Hip', 'Sagittal', 'Excursion', 'Hip extension excursion during stance', DEG),
    ('HIP_EXT_at_HS', 'Hip', 'Sagittal', 'Angle at initial contact', 'Hip flexion/extension angle at initial contact', DEG),
    ('HIP_EXT_percent_STANCE', 'Hip', 'Sagittal', 'Timing of peak angle', 'Timing of peak hip extension', PST),
    ('HIP_ADD_PEAK_ANGLE', 'Hip', 'Frontal', 'Peak angle', 'Peak hip adduction', DEG),
    ('HIP_ADD_EXCURSION', 'Hip', 'Frontal', 'Excursion', 'Hip adduction excursion during stance', DEG),
    ('HIP_ADD_at_HS', 'Hip', 'Frontal', 'Angle at initial contact', 'Hip adduction angle at initial contact', DEG),
    ('HIP_ADD_percent_STANCE', 'Hip', 'Frontal', 'Timing of peak angle', 'Timing of peak hip adduction', PST),
    ('PEAK_HIP_ADD_VELOCITY', 'Hip', 'Frontal', 'Peak angular velocity', 'Peak hip adduction velocity', DPS),
    ('HIP_ABD_PEAK_VEL', 'Hip', 'Frontal', 'Peak angular velocity', 'Peak hip abduction velocity', DPS),
    ('HIP_ABD_VEL_percent_STANCE', 'Hip', 'Frontal', 'Timing of peak angular velocity', 'Timing of peak hip abduction velocity', PST),
    ('HIP_ROT_PEAK_ANGLE', 'Hip', 'Transverse', 'Peak angle', 'Peak hip rotation', DEG),
    ('HIP_ROT_EXCURSION', 'Hip', 'Transverse', 'Excursion', 'Hip rotation excursion during stance', DEG),
    ('HIP_ROT_at_HS', 'Hip', 'Transverse', 'Angle at initial contact', 'Hip rotation angle at initial contact', DEG),
    ('HIP_ROT_percent_STANCE', 'Hip', 'Transverse', 'Timing of peak angle', 'Timing of peak hip rotation', PST),
    ('HIP_ROT_PEAK_VEL', 'Hip', 'Transverse', 'Peak angular velocity', 'Peak hip rotation velocity', DPS),
    # knee
    ('KNEE_FLEX_PEAK_ANGLE', 'Knee', 'Sagittal', 'Peak angle', 'Peak knee flexion', DEG),
    ('KNEE_FLEX_EXCURSION', 'Knee', 'Sagittal', 'Excursion', 'Knee flexion excursion during stance', DEG),
    ('KNEE_FLEX_at_HS', 'Knee', 'Sagittal', 'Angle at initial contact', 'Knee flexion angle at initial contact', DEG),
    ('KNEE_FLEX_percent_STANCE', 'Knee', 'Sagittal', 'Timing of peak angle', 'Timing of peak knee flexion', PST),
    ('KNEE_FLEX_PEAK_VEL', 'Knee', 'Sagittal', 'Peak angular velocity', 'Peak knee flexion velocity', DPS),
    ('KNEE_FLEX_VEL_percent_STANCE', 'Knee', 'Sagittal', 'Timing of peak angular velocity', 'Timing of peak knee flexion velocity', PST),
    ('KNEE_ADD_PEAK_ANGLE', 'Knee', 'Frontal', 'Peak angle', 'Peak knee adduction', DEG),
    ('KNEE_ADD_EXCURSION', 'Knee', 'Frontal', 'Excursion', 'Knee adduction excursion during stance', DEG),
    ('KNEE_ADD_at_HS', 'Knee', 'Frontal', 'Angle at initial contact', 'Knee adduction angle at initial contact', DEG),
    ('KNEE_ADD_percent_STANCE', 'Knee', 'Frontal', 'Timing of peak angle', 'Timing of peak knee adduction', PST),
    ('KNEE_ADD_PEAK_VEL', 'Knee', 'Frontal', 'Peak angular velocity', 'Peak knee adduction velocity', DPS),
    ('KNEE_ADD_VEL_percent_STANCE', 'Knee', 'Frontal', 'Timing of peak angular velocity', 'Timing of peak knee adduction velocity', PST),
    ('KNEE_ABD_PEAK_ANGLE', 'Knee', 'Frontal', 'Peak angle', 'Peak knee abduction', DEG),
    ('KNEE_ABD_EXCURSION', 'Knee', 'Frontal', 'Excursion', 'Knee abduction excursion during stance', DEG),
    ('KNEE_ABD_at_HS', 'Knee', 'Frontal', 'Angle at initial contact', 'Knee abduction angle at initial contact', DEG),
    ('KNEE_ABD_percent_STANCE', 'Knee', 'Frontal', 'Timing of peak angle', 'Timing of peak knee abduction', PST),
    ('KNEE_ABD_PEAK_VEL', 'Knee', 'Frontal', 'Peak angular velocity', 'Peak knee abduction velocity', DPS),
    ('KNEE_ABD_VEL_percent_STANCE', 'Knee', 'Frontal', 'Timing of peak angular velocity', 'Timing of peak knee abduction velocity', PST),
    ('KNEE_ROT_PEAK_ANGLE', 'Knee', 'Transverse', 'Peak angle', 'Peak knee rotation', DEG),
    ('KNEE_ROT_EXCURSION', 'Knee', 'Transverse', 'Excursion', 'Knee rotation excursion during stance', DEG),
    ('KNEE_ROT_at_HS', 'Knee', 'Transverse', 'Angle at initial contact', 'Knee rotation angle at initial contact', DEG),
    ('KNEE_ROT_percent_STANCE', 'Knee', 'Transverse', 'Timing of peak angle', 'Timing of peak knee rotation', PST),
    ('KNEE_ROT_PEAK_VEL', 'Knee', 'Transverse', 'Peak angular velocity', 'Peak knee rotation velocity', DPS),
    # ankle
    ('ANKLE_DF_PEAK_ANGLE', 'Ankle', 'Sagittal', 'Peak angle', 'Peak ankle dorsiflexion', DEG),
    ('ANKLE_DF_EXCURSION', 'Ankle', 'Sagittal', 'Excursion', 'Ankle dorsiflexion excursion during stance', DEG),
    ('ANKLE_DF_at_HS', 'Ankle', 'Sagittal', 'Angle at initial contact', 'Ankle dorsiflexion angle at initial contact', DEG),
    ('ANKLE_DF_percent_STANCE', 'Ankle', 'Sagittal', 'Timing of peak angle', 'Timing of peak ankle dorsiflexion', PST),
    ('ANKLE_DF_PEAK_VEL', 'Ankle', 'Sagittal', 'Peak angular velocity', 'Peak ankle dorsiflexion velocity', DPS),
    ('ANKLE_DF_VEL_percent_STANCE', 'Ankle', 'Sagittal', 'Timing of peak angular velocity', 'Timing of peak ankle dorsiflexion velocity', PST),
    ('ANKLE_EVE_PEAK_ANGLE', 'Ankle', 'Frontal', 'Peak angle', 'Peak ankle eversion', DEG),
    ('ANKLE_EVE_EXCURSION', 'Ankle', 'Frontal', 'Excursion', 'Ankle eversion excursion during stance', DEG),
    ('ANKLE_EVE_at_HS', 'Ankle', 'Frontal', 'Angle at initial contact', 'Ankle eversion angle at initial contact', DEG),
    ('ANKLE_EVE_percent_STANCE', 'Ankle', 'Frontal', 'Timing of peak angle', 'Timing of peak ankle eversion', 'fraction of stance (0–1)'),
    ('ANKLE_EVE_PEAK_VEL', 'Ankle', 'Frontal', 'Peak angular velocity', 'Peak ankle eversion velocity', DPS),
    ('ANKLE_EVE_VEL_percent_STANCE', 'Ankle', 'Frontal', 'Timing of peak angular velocity', 'Timing of peak ankle eversion velocity', PST),
    ('ANKLE_ROT_PEAK_ANGLE', 'Ankle', 'Transverse', 'Peak angle', 'Peak ankle rotation', DEG),
    ('ANKLE_ROT_EXCURSION', 'Ankle', 'Transverse', 'Excursion', 'Ankle rotation excursion during stance', DEG),
    ('ANKLE_ROT_at_HS', 'Ankle', 'Transverse', 'Angle at initial contact', 'Ankle rotation angle at initial contact', DEG),
    ('ANKLE_ROT_percent_STANCE', 'Ankle', 'Transverse', 'Timing of peak angle', 'Timing of peak ankle rotation', PST),
    ('ANKLE_ROT_PEAK_VEL', 'Ankle', 'Transverse', 'Peak angular velocity', 'Peak ankle rotation velocity', DPS),
    ('ANKLE_ROT_VEL_percent_STANCE', 'Ankle', 'Transverse', 'Timing of peak angular velocity', 'Timing of peak ankle rotation velocity', PST),
    # foot
    ('FOOT_PROG_ANGLE', 'Foot', 'Transverse', 'Foot', 'Foot progression angle', DEG),
    ('FOOT_ANG_at_HS', 'Foot', 'Sagittal', 'Foot', 'Foot angle at initial contact (foot strike angle)', DEG),
    ('FOOT_ANG_at_TO', 'Foot', 'Sagittal', 'Foot', 'Foot angle at toe-off', DEG),
    ('PRONATION_ONSET', 'Foot', 'Frontal', 'Foot', 'Onset of pronation', PST),
    ('PRONATION_OFFSET', 'Foot', 'Frontal', 'Foot', 'End of pronation', PST),
    ('MED_HEEL_WHIP_PEAK', 'Foot', 'Transverse', 'Foot', 'Peak medial heel whip', DEG),
    ('MHW_percent_SWING', 'Foot', 'Transverse', 'Foot', 'Timing of peak medial heel whip', '% swing'),
    ('MHW_EXC_FROM_TO', 'Foot', 'Transverse', 'Foot', 'Medial heel whip excursion from toe-off', DEG),
]
assert len(VARIABLES) == 76 and len({v[0] for v in VARIABLES}) == 76

if __name__ == '__main__':
    import pandas as pd
    data, out = sys.argv[1], sys.argv[2]
    run = pd.read_csv(os.path.join(data, 'ml_run_dataset.csv')).dropna()
    walk = pd.read_csv(os.path.join(data, 'ml_walk_dataset.csv')).dropna()
    rows = []
    for name, region, plane, vtype, desc, unit in VARIABLES:
        const = run[name].nunique() == 1 and walk[name].nunique() == 1 and run[name].iloc[0] == 0 and walk[name].iloc[0] == 0
        fmt = lambda s: '0 in every trial' if const else f"{s.mean():.2f} ± {s.std():.2f}"
        rows.append(dict(Variable=name, Region=region, Plane=plane, Type=vtype, Description=desc, Unit=unit,
                         Running=fmt(run[name]), Walking=fmt(walk[name]), Constant=const))
    T = pd.DataFrame(rows)
    assert set(c for c in run.columns if c not in ('sub_id', 'Target', 'Group', 'Age', 'Mass', 'Height', 'Sex', 'Speed')) == set(T.Variable)
    os.makedirs(out, exist_ok=True); T.to_csv(os.path.join(out, 'TableS2_variables.csv'), index=False)
    print(T.groupby('Constant').size().to_string()); print(T.to_string(index=False, max_colwidth=40))
