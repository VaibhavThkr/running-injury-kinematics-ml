"""Fig. 1 — analysis workflow (drawn programmatically). Counts match Section 2.1 and Fig. 2.
usage: python3 make_fig1.py <out_dir>
"""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = sys.argv[1]
N_FINAL, RUN_P, RUN_T, WALK_P, WALK_T, BOTH = 393, 366, 434, 350, 391, 323

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
W, H = 7.0, 4.30
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

EDGE, LW, FILL, WHITE = "#222222", 0.9, "#EEF2F6", "#FFFFFF"
WALK, RUN = "#2a78d6", "#e34948"
STEP = 0.155


def box(x0, y0, w, h, lines, fill=FILL, accent=None, va_top=None, step=None):
    """lines: (text, size, weight, style, align). Text block is vertically centred unless va_top is given."""
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fill, ec=EDGE, lw=LW))
    if accent:
        ax.add_patch(Rectangle((x0 + 0.012, y0 + 0.03), 0.055, h - 0.06, fc=accent, ec="none"))
    st_ = step or STEP
    steps = [0.6 * st_ if t == "" else st_ for (t, *_) in lines[1:]]   # blank lines are 0.6 of a line
    ytop = (y0 + h / 2 + sum(steps) / 2) if va_top is None else y0 + h - va_top
    y = ytop
    for i, (t, sz, wt, st, al) in enumerate(lines):
        if i: y -= steps[i - 1]
        x = x0 + w / 2 if al == "center" else x0 + 0.12 + (0.04 if accent else 0)
        if t: ax.text(x, y, t, ha=al, va="center", fontsize=sz, fontweight=wt, fontstyle=st)
    return dict(l=x0, r=x0 + w, b=y0, t=y0 + h, cx=x0 + w / 2, cy=y0 + h / 2)


def T(t, sz=7.5, wt="normal", st="normal", al="center"): return (t, sz, wt, st, al)


def arrow(x0, y0, x1, y1, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=LW, color=EDGE, mutation_scale=9, shrinkA=0, shrinkB=0, linestyle=ls))


def line(xs, ys): ax.plot(xs, ys, color=EDGE, lw=LW, solid_capstyle="butt")


# ---------------- top row: data -> modalities -> features -> models ----------------
BW, GAP, X0 = 1.52, 0.25, 0.08
xs = [X0 + i * (BW + GAP) for i in range(4)]
TOP_B, TOP_H = 2.55, 1.62

b1 = box(xs[0], TOP_B, BW, TOP_H, [
    T("Running Injury Clinic", 8, "bold"), T("Kinematic Dataset", 8, "bold"), T(""),
    T(f"{N_FINAL} participants with a", 7.5), T("primary diagnosis of", 7.5),
    T("PFPS, ITBS, plantar fasciitis,", 7), T("Achilles tendinopathy or MTSS", 7),
])

MH = 0.70
bw = box(xs[1], TOP_B + TOP_H - MH, BW, MH, [
    T("Walking trials", 8, "bold", al="left"), T(f"{WALK_T} trials", al="left"), T(f"{WALK_P} participants", al="left"),
], fill=WHITE, accent=WALK)
br = box(xs[1], TOP_B, BW, MH, [
    T("Running trials", 8, "bold", al="left"), T(f"{RUN_T} trials", al="left"), T(f"{RUN_P} participants", al="left"),
], fill=WHITE, accent=RUN)

b3 = box(xs[2], TOP_B, BW, TOP_H, [
    T("Feature extraction", 8, "bold"), T("(each modality separately)", 7, st="italic"), T(""),
    T("76 discrete gait", 7.5), T("variables per trial", 7.5), T("(kinematic, spatiotemporal)", 7.2), T(""),
    T("Session mean over strides;", 7), T("injured limb, or the mean", 7), T("of both limbs", 7),
])

b4 = box(xs[3], TOP_B, BW, TOP_H, [
    T("XGBoost classifiers", 8, "bold"), T("(each modality separately)", 7, st="italic"), T(""),
    T("5-way multi-class model", 7.5), T("5 one-versus-rest models", 7.5), T(""),
    T("Fixed hyperparameters:", 7), T("100 trees, depth 3,", 7), T("learning rate 0.1", 7),
])

# arrows, top row
ymid = TOP_B + TOP_H / 2
xm = (b1["r"] + bw["l"]) / 2
line([b1["r"], xm], [ymid, ymid]); line([xm, xm], [br["cy"], bw["cy"]])
arrow(xm, bw["cy"], bw["l"], bw["cy"]); arrow(xm, br["cy"], br["l"], br["cy"])
xm2 = (bw["r"] + b3["l"]) / 2
line([bw["r"], xm2], [bw["cy"], bw["cy"]]); line([br["r"], xm2], [br["cy"], br["cy"]]); line([xm2, xm2], [br["cy"], bw["cy"]])
arrow(xm2, ymid, b3["l"], ymid)
arrow(b3["r"], ymid, b4["l"], ymid)

# ---------------- bottom row: validation -> evaluation ; sensitivity ----------------
BOT_B, BOT_H = 0.10, 2.05
b5 = box(xs[3], BOT_B, BW, BOT_H, [
    T("Participant-grouped", 8, "bold"), T("5-fold cross-validation", 8, "bold"), T(""),
    T("All trials of a participant", 7.5), T("in the same fold", 7.5), T(""),
    T("Scaling and class weights", 7), T("estimated from the", 7), T("training folds only", 7),
])
arrow(b4["cx"], b4["b"], b4["cx"], b5["t"])

EW = 2 * BW + GAP
b6 = box(xs[1], BOT_B, EW, BOT_H, [T("Evaluation of pooled out-of-fold predictions", 8, "bold", al="left")], va_top=0.17)
LABX, TXTX, y = b6["l"] + 0.12, b6["l"] + 1.20, b6["t"] - 0.17 - 1.6 * STEP
for lab, rows in [("One-versus-rest", ["ROC AUC with 95% CIs from 2000",
                                       "participant-level bootstrap resamples,",
                                       "and average precision; permutation test",
                                       "against chance (1000 permutations,",
                                       "Holm-adjusted)"]),
                  ("Multi-class", ["Accuracy, balanced accuracy and macro-F1,",
                                   "compared with the majority-class rate"]),
                  ("Walking vs running", ["Paired participant-level bootstrap of the",
                                          "AUC difference, Holm-adjusted"])]:
    ax.text(LABX, y, lab, ha="left", va="center", fontsize=7.0, fontweight="bold")
    for r_ in rows:
        ax.text(TXTX, y, r_, ha="left", va="center", fontsize=7.0); y -= STEP
    y -= 0.35 * STEP
arrow(b5["l"], b5["cy"], b6["r"], b5["cy"])

b7 = box(xs[0], BOT_B, BW, BOT_H, [
    T("Sensitivity analyses", 8, "bold"), T(""),
    T("• Trial-level cross-validation", 6.8, al="left"), T("• 20 random fold assignments", 6.8, al="left"),
    T("• 1 session per participant", 6.8, al="left"), T("• No stride length or rate", 6.8, al="left"),
    T("• No spatiotemporal variables", 6.8, al="left"), T("• Speed-residualized variables", 6.8, al="left"),
    T("• Speed added as a predictor", 6.8, al="left"), T("• Speed, age, mass and height", 6.8, al="left"),
    T("   added as predictors", 6.8, al="left"), T("• Age, mass and height only", 6.8, al="left"),
    T(f"• {BOTH} participants with", 6.8, al="left"), T("   both modalities", 6.8, al="left"),
], fill=WHITE, step=0.138)
arrow(b6["l"], b5["cy"], b7["r"], b5["cy"], ls=(0, (3, 2)))

os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf", "tif"):
    kw = dict(dpi=600) if ext in ("png", "tif") else {}
    if ext == "tif": kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(os.path.join(OUT, f"Fig1_workflow.{ext}"), facecolor="white", **kw)
print("saved")
