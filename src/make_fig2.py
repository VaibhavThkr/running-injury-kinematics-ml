"""Fig. 2 — participant flow diagram (drawn programmatically). usage: python make_fig2.py <out_dir>
All counts come from the authors' re-audit of run_data_meta.csv / walk_data_meta.csv and the final ML datasets.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox

# ---------------- counts (single source of truth) ----------------
N_DB, N_RUN_REC, N_WALK_REC = 1798, 1832, 2088
N_EXCL_DEMO = 2
N_DEMO = N_DB - N_EXCL_DEMO                    # 1796
N_EXCL_TARGET = 1403
N_FINAL = N_DEMO - N_EXCL_TARGET               # 393
GROUPS = [("PFPS", 137), ("ITBS", 100), ("Plantar fasciitis", 54), ("Achilles tendinopathy", 52), ("MTSS", 50)]
RUN_P, RUN_T, WALK_P, WALK_T = 366, 434, 350, 391
BOTH, RUN_ONLY, WALK_ONLY = 323, 43, 27
assert N_FINAL == 393 == sum(n for _, n in GROUPS) == BOTH + RUN_ONLY + WALK_ONLY
assert RUN_P == BOTH + RUN_ONLY and WALK_P == BOTH + WALK_ONLY
# trials per participant (author's re-audit, 2026-10-27) and session records dropped at feature extraction
RUN_DIST = {1: 329, 2: 18, 3: 9, 4: 8, 5: 2}
WALK_DIST = {1: 323, 2: 17, 3: 7, 4: 2, 5: 1}
RUN_ELIG, RUN_EXCL_REC, WALK_ELIG, WALK_EXCL_REC = 438, 4, 391, 0
for dist, p, t in [(RUN_DIST, RUN_P, RUN_T), (WALK_DIST, WALK_P, WALK_T)]:
    assert sum(dist.values()) == p and sum(k * v for k, v in dist.items()) == t
assert RUN_ELIG - RUN_EXCL_REC == RUN_T and WALK_ELIG - WALK_EXCL_REC == WALK_T

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})
W, H = 7.0, 7.2
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.set_in_layout(False)

EDGE, LW = "#222222", 0.9
MAIN_FILL, EXCL_FILL = "#EEF2F6", "#FFFFFF"

def box(xc, yc, w, h, lines, fill):
    ax.add_patch(FancyBboxPatch((xc - w/2, yc - h/2), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=fill, ec=EDGE, lw=LW))
    # lines: list of (text, size, weight, style)
    n = len(lines); step = 0.165
    y0 = yc + (n - 1) * step / 2
    for i, (t, sz, wt, st, ha) in enumerate(lines):
        x = xc if ha == "center" else xc - w/2 + 0.12
        ax.text(x, y0 - i*step, t, ha=ha, va="center", fontsize=sz, fontweight=wt, fontstyle=st)
    return (xc - w/2, xc + w/2, yc - h/2, yc + h/2)

def L(t, sz=8, wt="normal", st="normal", ha="center"): return (t, sz, wt, st, ha)

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=LW, color=EDGE, mutation_scale=9, shrinkA=0, shrinkB=0))

def line(xs, ys): ax.plot(xs, ys, color=EDGE, lw=LW, solid_capstyle="butt")

MX, MW = 2.45, 3.9          # main column centre / width
EX, EW = 5.75, 2.25         # exclusion column centre / width

# Box 1 — source database
b1 = box(MX, 6.55, MW, 0.78, [
    L("Running Injury Clinic Kinematic Dataset (Ferber et al., 2024)", 8.5, "bold"),
    L(f"n = {N_DB} participants"),
    L(f"({N_RUN_REC} running and {N_WALK_REC} walking session records)", 7.5, st="italic"),
], MAIN_FILL)

# Box 2 — complete data
b2 = box(MX, 5.05, MW, 0.62, [
    L("Complete demographic and gait-speed data", 8.5, "bold"),
    L(f"n = {N_DEMO} participants"),
], MAIN_FILL)

# Box 3 — final cohort
b3 = box(MX, 3.2, MW, 1.12, [
    L("Final analyzed cohort", 8.5, "bold"),
    L(f"n = {N_FINAL} participants with a target primary diagnosis"),
    L("PFPS (n = 137)  ·  ITBS (n = 100)  ·  plantar fasciitis (n = 54)", 7.5),
    L("Achilles tendinopathy (n = 52)  ·  MTSS (n = 50)", 7.5),
], MAIN_FILL)

# exclusion boxes
yE1 = (b1[2] + b2[3]) / 2
e1 = box(EX, yE1, EW, 0.55, [
    L(f"Excluded: n = {N_EXCL_DEMO} participants", 8, "bold", ha="left"),
    L("Missing demographic or gait-speed data", 7.5, ha="left"),
], EXCL_FILL)
yE2 = (b2[2] + b3[3]) / 2
e2 = box(EX, yE2, EW, 0.9, [
    L(f"Excluded: n = {N_EXCL_TARGET} participants", 8, "bold", ha="left"),
    L("• Healthy (pain-free) controls", 7.5, ha="left"),
    L("• Primary diagnosis non-target or", 7.5, ha="left"),
    L("   unspecified (including osteoarthritis)", 7.5, ha="left"),
], EXCL_FILL)

# main arrows + branches
arrow(MX, b1[2], MX, b2[3]); line([MX, e1[0]], [yE1, yE1])
arrow(MX, b2[2], MX, b3[3]); line([MX, e2[0]], [yE2, yE2])
# make branch arrowheads visible
ax.annotate("", xy=(e1[0], yE1), xytext=(e1[0] - 0.25, yE1), arrowprops=dict(arrowstyle="-|>", lw=LW, color=EDGE, mutation_scale=9, shrinkA=0, shrinkB=0))
ax.annotate("", xy=(e2[0], yE2), xytext=(e2[0] - 0.25, yE2), arrowprops=dict(arrowstyle="-|>", lw=LW, color=EDGE, mutation_scale=9, shrinkA=0, shrinkB=0))

# split into modality analyses
RX, WX, BW = 1.85, 5.15, 2.95
ysplit = b3[2] - 0.30
line([MX, MX], [b3[2], ysplit]); line([RX, WX], [ysplit, ysplit])
yb = 1.62
r = box(RX, yb, BW, 0.86, [
    L("Running analysis", 8.5, "bold"),
    L(f"{RUN_P} participants  ·  {RUN_T} trials"),
    L(f"{RUN_DIST[1]} with 1 trial  ·  {RUN_P - RUN_DIST[1]} with 2–5 trials", 7.5),
    L(f"{RUN_EXCL_REC} session records excluded: incomplete kinematic data", 7, st="italic"),
], MAIN_FILL)
w = box(WX, yb, BW, 0.86, [
    L("Walking analysis", 8.5, "bold"),
    L(f"{WALK_P} participants  ·  {WALK_T} trials"),
    L(f"{WALK_DIST[1]} with 1 trial  ·  {WALK_P - WALK_DIST[1]} with 2–5 trials", 7.5),
    L("No session records excluded", 7, st="italic"),
], MAIN_FILL)
arrow(RX, ysplit, RX, r[3]); arrow(WX, ysplit, WX, w[3])

ax.text(W/2, 0.78, f"{BOTH} participants contributed both running and walking trials; {RUN_ONLY} running only; {WALK_ONLY} walking only.",
        ha="center", va="center", fontsize=7.5)
ax.text(W/2, 0.55, "Counts are unique participants unless stated otherwise. A trial is one treadmill session recording.",
        ha="center", va="center", fontsize=7.5, fontstyle="italic", color="#444444")

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf", "tif"):
    kw = dict(dpi=600) if ext in ("png", "tif") else {}
    if ext == "tif": kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(os.path.join(OUT, f"Fig2_participant_flow.{ext}"), facecolor="white", bbox_inches=Bbox([[0.31, 0.44], [6.94, 7.01]]), **kw)
print("saved")
