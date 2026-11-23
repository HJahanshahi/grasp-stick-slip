# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""fig3_law_schematic.py - Figure 3: the force-scaled asymmetric stick-slip law.
(a) three elasto-slip elements in parallel between the hand and the rim; (b) torque of one element against rim angle.
Run: python3 fig3_law_schematic.py -> out/figs/fig3_law_schematic.pdf/.png"""
import os, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
warnings.filterwarnings("ignore", message="findfont")
plt.rcParams.update({"font.family": "Inter", "mathtext.fontset": "dejavusans", "font.size": 8,
                     "savefig.dpi": 600, "pdf.fonttype": 42, "axes.linewidth": 0.6})
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "out", "figs")
INK, GREY, BLUE, ORANGE = "#1a1a1a", "#6b6b6b", "#0072B2", "#D55E00"

fig = plt.figure(figsize=(7.2, 2.75))
# ---------------- (a) network
ax = fig.add_axes([0.0, 0.02, 0.50, 0.92]); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off"); ax.set_aspect("equal")
# hand (fixed wall, left) and rim (moving bar, right)
ax.add_patch(Rectangle((6, 6), 3, 44, fc="#d9d9d9", ec=INK, lw=0.6))
for y in np.arange(7, 50, 3): ax.plot([4, 6], [y - 1.5, y], color=INK, lw=0.5)
ax.text(7.5, 52.5, "hand", ha="center", fontsize=8)
ax.add_patch(Rectangle((84, 6), 3, 44, fc="#f2e3b3", ec=INK, lw=0.6))
ax.text(85.5, 52.5, "rim", ha="center", fontsize=8)
ax.add_patch(FancyArrowPatch((90, 28), (98, 28), arrowstyle="-|>", mutation_scale=8, color=INK, lw=0.8))
ax.text(94, 30.5, r"$\theta$", ha="center", fontsize=9)

def spring(x0, x1, y, n=6, amp=1.6):
    xs = np.linspace(x0 + 2, x1 - 2, 2 * n + 1); ys = y + amp * np.array([0] + [(-1) ** i for i in range(2 * n - 1)] + [0])
    ax.plot([x0] + list(xs) + [x1], [y] + list(ys) + [y], color=INK, lw=0.7)
rows = [(40, "1"), (26, "2"), (12, "3")]
for y, j in rows:
    ax.plot([9, 14], [y, y], color=INK, lw=0.7)
    spring(14, 38, y)
    ax.text(26, y + 3.0, rf"$s\,k_{j}$", ha="center", fontsize=8)
    # slider block resting on a plate fixed to the rim; the block sticks until its torque reaches a limit
    ax.plot([38, 48], [y, y], color=INK, lw=0.7)
    ax.add_patch(Rectangle((48, y - 2.2), 8, 4.4, fc="white", ec=INK, lw=0.7))
    ax.add_patch(Rectangle((44, y - 3.6), 40, 1.4, fc="#f2e3b3", ec=INK, lw=0.6))
    ax.text(58, y + 0.2, rf"$-s\,c_{j}^{{+}} \leq e_{j} \leq s\,c_{j}^{{-}}$", ha="left", fontsize=7, va="center")
ax.text(46, 0.8, r"$\tau = e_1 + e_2 + e_3$,   $s = F / F_{\mathrm{ref}}$", ha="center", fontsize=8)
ax.text(0, 57, "(a)", fontsize=8, fontweight="bold")

# ---------------- (b) one element against rim angle
bx = fig.add_axes([0.58, 0.18, 0.40, 0.72])
k, cp, cn = 1.0, 0.45, 1.0      # schematic values: slip limit smaller than wedging limit
def run(s, path):
    e, th, E, T = 0.0, path[0], [0.0], [path[0]]
    for t1 in path[1:]:
        for t in np.linspace(th, t1, 400)[1:]:
            e = np.clip(e - s * k * (t - th), -s * cp, s * cn); th = t; E.append(e); T.append(t)
    return np.array(T), np.array(E)
path = [0, -1.8, 1.4, -1.8]
for s, ls, c in [(1.0, "-", BLUE), (0.6, "--", BLUE)]:
    T, E = run(s, path); bx.plot(T, E, ls, color=c, lw=1.2 if s == 1 else 0.9)
bx.axhline(0, color=GREY, lw=0.5); bx.axvline(0, color=GREY, lw=0.5)
bx.set_xlim(-2.0, 1.6); bx.set_ylim(-0.75, 1.25)
bx.set_xticks([]); bx.set_yticks([])
for sp in ("top", "right"): bx.spines[sp].set_visible(False)
bx.set_xlabel(r"rim angle $\theta$", labelpad=2); bx.set_ylabel(r"element torque $e_j$", labelpad=2)
bx.text(-1.95, 1.08, r"$s\,c_j^{-}$  (wedging limit)", fontsize=7, va="center")
bx.text(1.55, -0.53, r"$-s\,c_j^{+}$  (slip limit)", fontsize=7, va="center", ha="right")
bx.annotate(r"slope $-s\,k_j$ (stick)", xy=(-0.40, 0.40), xytext=(0.15, 0.80), fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY))
bx.text(-1.95, -0.25, "dashed: lower $F$,\nsame corner angles,\nsmaller torques", fontsize=6.5, color=BLUE, va="top")
bx.text(-0.06, -0.1, "wedging", fontsize=7, ha="right", color=GREY); bx.text(0.06, -0.1, "slip", fontsize=7, ha="left", color=GREY)
bx.text(-0.16, 1.04, "(b)", transform=bx.transAxes, fontsize=8, fontweight="bold")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, "fig3_law_schematic." + ext), bbox_inches="tight", pad_inches=0.04)
print("ok")
