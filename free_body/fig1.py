# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""fig1.py - Fig. 1 drawn from scratch. Panels (c) and (d) are computed from the fitted law (Table 1), not sketched."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "..", "figures"); os.makedirs(FIG, exist_ok=True)
sys.argv = ["x", "none", "1"]; sys.path.insert(0, HERE)
import law_free_body as L
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, FancyBboxPatch, Circle, Wedge, Arc, FancyArrowPatch, ConnectionPatch, Rectangle
plt.rcParams.update({"font.family": "Inter", "font.size": 8, "savefig.dpi": 600, "mathtext.fontset": "custom", "mathtext.it": "Inter:italic", "mathtext.rm": "Inter"})
K = "#1c1c1c"; GOLD, GOLD_T, GOLD_S = "#D9B24C", "#EBD282", "#B48F30"; CONE, CONE_D, RIMC = "#C9CED6", "#9AA1AD", "#ECEEF1"; PAD, PAD_E = "#3F8052", "#1F4A2C"
HAND, JNT = "#2B2B2B", "#8A8A8A"; ARM, ARM_E = "#B9C0CB", "#6F7682"; CW, CS, G = "#B9561C", "#5B3A94", "#8C8C8C"; CO, CB = "#D55E00", "#0072B2"
fig = plt.figure(figsize=(7.2, 4.7))
def tag(ax, s, title, dx=0.0):          # panel letter only; what each panel shows is said in the caption
    ax.text(dx, 1.02, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom", ha="left")

# ------------------------------------------------------------------ (a) scene
A = fig.add_axes([0.005, 0.445, 0.635, 0.50]); A.set_xlim(0, 10.9); A.set_ylim(0.2, 5.75); A.set_anchor("NW"); A.set_aspect("equal"); A.axis("off")
P = lambda pts, fc, z=2, **k: A.add_patch(Polygon(pts, closed=True, fc=fc, ec=K, lw=0.8, zorder=z, joinstyle="round", **k))
P([(0.2, 0.5), (3.0, 0.5), (3.0, 4.0), (0.2, 4.0)], GOLD); P([(0.2, 4.0), (3.0, 4.0), (4.0, 4.75), (1.2, 4.75)], GOLD_T); P([(3.0, 0.5), (4.0, 1.25), (4.0, 4.75), (3.0, 4.0)], GOLD_S)
A.text(1.6, 0.85, "spinning target", ha="center", va="center", fontsize=8)
cy = 2.62; tx, rx, a, b = 3.55, 6.0, 0.40, 1.30
A.plot([2.9, 7.6], [cy, cy], color=K, lw=0.5, ls=(0, (6, 2, 1, 2)), zorder=1.5)
P([(tx, cy + 0.30), (rx, cy + b), (rx, cy - b), (tx, cy - 0.30)], CONE, z=3)
A.add_patch(Ellipse((tx, cy), 0.20, 0.60, fc=CONE_D, ec=K, lw=0.8, zorder=3.1))
A.add_patch(Ellipse((rx, cy), 2 * a, 2 * b, fc=RIMC, ec=K, lw=0.9, zorder=3.2)); A.add_patch(Ellipse((rx + 0.02, cy), 2 * a * 0.80, 2 * b * 0.90, fc=CONE_D, ec=K, lw=0.5, zorder=3.3))
A.plot([rx, 7.6], [cy, cy], color=K, lw=0.5, ls=(0, (6, 2, 1, 2)), zorder=3.4)
tt = np.radians(np.linspace(112, -112, 80)); sx, sy = 4.75 + 0.30 * np.cos(tt), cy + 1.02 * np.sin(tt)
A.plot(sx, sy, color=K, lw=1.3, zorder=4, solid_capstyle="round")
A.add_patch(FancyArrowPatch((sx[-4], sy[-4]), (sx[-1] - 0.035, sy[-1] + 0.012), arrowstyle="-|>", mutation_scale=10, color=K, lw=0, zorder=4))
A.text(4.30, cy + 1.30, r"$\omega$", fontsize=11, ha="center", va="center")
# hand: palm, two outer fingers over the lip, one inner finger inside the opening
pads_out = [(5.75, 3.925), (6.25, 3.925)]; pad_in = (6.0, 3.715); r = 0.135
def finger(pts, z):
    xs, ys = zip(*pts); A.plot(xs, ys, color=K, lw=6.2, solid_capstyle="round", solid_joinstyle="round", zorder=z); A.plot(xs, ys, color=HAND, lw=5.0, solid_capstyle="round", solid_joinstyle="round", zorder=z + 0.01)
    for x, y in pts[1:-1]: A.add_patch(Circle((x, y), 0.075, fc=JNT, ec=K, lw=0.5, zorder=z + 0.02))
finger([(7.75, 3.75), (7.0, 3.30), (6.38, 3.36), (pad_in[0] + 0.07, pad_in[1] - 0.02)], 5)
finger([(7.75, 4.55), (7.05, 4.95), (6.22, 4.72), (pads_out[0][0] + 0.02, pads_out[0][1] + 0.05)], 6)
finger([(7.75, 4.35), (7.15, 4.62), (6.55, 4.42), (pads_out[1][0] + 0.03, pads_out[1][1] + 0.04)], 7)
for (x, y), z in ((pad_in, 5.5), (pads_out[0], 6.5), (pads_out[1], 7.5)): A.add_patch(Circle((x, y), r, fc=PAD, ec=PAD_E, lw=0.7, zorder=z))
A.add_patch(FancyBboxPatch((7.62, 3.52), 1.15, 1.28, boxstyle="round,pad=0.02,rounding_size=0.16", fc=HAND, ec=K, lw=0.8, zorder=8))
A.text(8.2, 5.02, "three-finger hand", ha="center", va="bottom", fontsize=7.5)
# arm and servicer
def link(p, q, z=7.5): A.plot(*zip(p, q), color=ARM_E, lw=6.0, solid_capstyle="round", zorder=z); A.plot(*zip(p, q), color=ARM, lw=4.6, solid_capstyle="round", zorder=z + 0.01)
link((8.75, 4.16), (9.45, 4.16)); link((9.45, 4.16), (10.0, 2.95), 7.2); link((10.0, 2.95), (9.75, 1.95), 7.0)
for x, y in ((9.45, 4.16), (10.0, 2.95)): A.add_patch(Circle((x, y), 0.20, fc=CONE_D, ec=K, lw=0.7, zorder=7.8))
P([(8.9, 0.55), (10.45, 0.55), (10.45, 1.85), (8.9, 1.85)], "#D4D8DE", z=8.2); P([(8.9, 1.85), (10.45, 1.85), (10.8, 2.12), (9.25, 2.12)], "#E6E8EC", z=8.1); P([(10.45, 0.55), (10.8, 0.82), (10.8, 2.12), (10.45, 1.85)], "#AEB4BE", z=8.1)
A.text(9.675, 1.2, "servicer", ha="center", va="center", fontsize=7.5, zorder=9)
zc = Circle((6.0, 3.86), 0.56, fc="none", ec=K, lw=0.7, ls=(0, (3, 2)), zorder=9); A.add_patch(zc)
A.annotate("(b)", xy=(6.0 - 0.56 * np.cos(np.radians(40)), 3.86 + 0.56 * np.sin(np.radians(40))), xytext=(5.15, 4.95), fontsize=7.5, ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=K, shrinkA=2, shrinkB=1))
tag(A, "(a)", "Capture of a spinning target by the rim of its cone", dx=0.01)

# ------------------------------------------------------------------ (b) rim lip seen along the cone axis
B = fig.add_axes([0.655, 0.485, 0.335, 0.445]); B.set_anchor("N"); B.set_xlim(-1.62, 1.62); B.set_ylim(-0.42, 1.92); B.set_aspect("equal"); B.set_xticks([]); B.set_yticks([])
for s_ in B.spines.values(): s_.set_linewidth(0.7); s_.set_color("#555555")
B.set_facecolor("#F6F7F9"); R = 1.2; wth = 0.085
B.add_patch(Wedge((0, 0), R + wth / 2, 12, 168, width=wth, fc=CONE, ec=K, lw=0.7, zorder=3))
def stub(ang, r0, r1, z=2):
    c, s_ = np.cos(np.radians(ang)), np.sin(np.radians(ang)); B.plot([r0 * c, r1 * c], [r0 * s_, r1 * s_], color=HAND, lw=7.5, solid_capstyle="butt", zorder=z)
rp = 0.13
for ang in (58, 122):
    c, s_ = np.cos(np.radians(ang)), np.sin(np.radians(ang)); rc = R + wth / 2 + rp; stub(ang, rc, rc + 0.36); B.add_patch(Circle((rc * c, rc * s_), rp, fc=PAD, ec=PAD_E, lw=0.7, zorder=4))
    ra = rc + 0.27; off = 0.20 * (1 if ang < 90 else -1); px, py = -s_ * off, c * off
    B.add_patch(FancyArrowPatch((ra * c + 0.20 * c - px, ra * s_ + 0.20 * s_ - py), ((rc - 0.02) * c + 0.02 * c - px, (rc - 0.02) * s_ + 0.02 * s_ - py), arrowstyle="-|>", mutation_scale=8, color=K, lw=1.0, zorder=5))
    B.text((ra + 0.33) * c - px * 1.25, (ra + 0.33) * s_ - py * 1.25, r"$F$", fontsize=9, ha="center", va="center")
rc = R - wth / 2 - rp; stub(90, rc - 0.36, rc); B.add_patch(Circle((0, rc), rp, fc=PAD, ec=PAD_E, lw=0.7, zorder=4))
B.add_patch(FancyArrowPatch((-0.24, rc - 0.42), (-0.24, rc - 0.02), arrowstyle="-|>", mutation_scale=8, color=K, lw=1.0, zorder=5)); B.text(-0.38, rc - 0.25, r"$F$", fontsize=9, ha="center", va="center")
B.add_patch(Arc((0, 0), 2 * 0.62, 2 * 0.62, theta1=38, theta2=142, color=K, lw=1.2, zorder=3)); t_ = np.radians(38)
B.add_patch(FancyArrowPatch((0.62 * np.cos(t_ + 0.10), 0.62 * np.sin(t_ + 0.10)), (0.62 * np.cos(t_ - 0.03), 0.62 * np.sin(t_ - 0.03)), arrowstyle="-|>", mutation_scale=9, color=K, lw=0, zorder=3))
B.text(0, 0.22, r"rim turns by $\theta$", ha="center", va="center", fontsize=7.5)
B.add_patch(Arc((0, 0), 2 * 0.90, 2 * 0.90, theta1=14, theta2=40, color=CW, lw=1.2, zorder=3)); t_ = np.radians(40)
B.add_patch(FancyArrowPatch((0.90 * np.cos(t_ - 0.10), 0.90 * np.sin(t_ - 0.10)), (0.90 * np.cos(t_ + 0.03), 0.90 * np.sin(t_ + 0.03)), arrowstyle="-|>", mutation_scale=9, color=CW, lw=0, zorder=3))
B.text(0.84, -0.13, "torque $\\tau$ on the body", ha="center", va="center", fontsize=7, color=CW)
B.add_patch(Circle((0, -0.13), 0.045, fc="none", ec=K, lw=0.7)); B.add_patch(Circle((0, -0.13), 0.012, fc=K, ec=K)); B.text(-0.09, -0.13, "cone axis", ha="right", va="center", fontsize=7, color="#444444")
B.annotate("outer wall", xy=(-(R + wth / 2) * np.cos(np.radians(28)), (R + wth / 2) * np.sin(np.radians(28))), xytext=(-1.55, 1.05), fontsize=6.5, color="#444444", ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color="#777777", shrinkA=1, shrinkB=1))
B.annotate("inner wall", xy=((R - wth / 2) * np.cos(np.radians(47)), (R - wth / 2) * np.sin(np.radians(47))), xytext=(0.34, 0.66), fontsize=6.5, color="#444444", ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color="#777777", shrinkA=1, shrinkB=1))
tag(B, "(b)", "", dx=0.0)
# ------------------------------------------------------------------ (c) the fitted law, drawn from its parameters
Pp = L.P; k, cp, cn = Pp[:3], Pp[3:6], Pp[6:9]
def loop(F, Aamp=15.0, n=600):
    s = F / L.F_REF; th = np.radians(np.concatenate([np.linspace(0, Aamp, n), np.linspace(Aamp, -Aamp, 2 * n), np.linspace(-Aamp, Aamp, 2 * n), np.linspace(Aamp, -Aamp, 2 * n)])); e = np.zeros(3); tau = []
    for i in range(len(th)):
        d = th[i] - th[i - 1] if i else 0.0; e = np.clip(e - s * k * d, -s * cp, s * cn); tau.append(e.sum())
    m = slice(3 * n, 7 * n + 1); return np.degrees(th[m]), np.array(tau)[m], s
C = fig.add_axes([0.075, 0.105, 0.385, 0.305])
for sp in ("top", "right"): C.spines[sp].set_visible(False)
C.axhline(0, color="#555555", lw=0.5); C.axvline(0, color="#555555", lw=0.5)
th, tau, s = loop(25.0); C.plot(th, tau, color=G, lw=1.1, ls=(0, (3, 2)))
th, tau, s = loop(50.0); stf = np.abs(np.gradient(tau, np.radians(th))); col = np.where(stf > 0.55 * s * k.sum(), 0, np.where(np.gradient(th) < 0, 1, 2))
for c_, cc in ((1, CW), (2, CS), (0, K)):
    idx = np.flatnonzero(col == c_)
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(seg) > 1: seg = np.r_[seg, min(seg[-1] + 1, len(th) - 1)]; C.plot(th[seg], tau[seg], color=cc, lw=2.0, solid_capstyle="round")
C.set_xlim(-19, 19); C.set_ylim(-1.9, 4.9); C.set_xticks([-15, -10, -5, 0, 5, 10, 15]); C.set_yticks([-1, 0, 1, 2, 3, 4]); C.tick_params(labelsize=7, width=0.6, length=2.5)
C.set_xlabel(r"rim angle $\theta$ (deg)", fontsize=7.5); C.set_ylabel(r"torque $\tau$ (N·m)", fontsize=7.5)
C.text(-7.0, 4.02, "wedging", color=CW, ha="center", va="bottom", fontsize=7.5, fontweight="semibold"); C.text(8.0, -1.25, "slip", color=CS, ha="center", va="top", fontsize=7.5, fontweight="semibold")
C.text(-14.4, 2.55, "stick", ha="right", va="center", fontsize=7.5); C.text(13.3, 1.0, "stick", ha="left", va="center", fontsize=7.5)
C.plot([3.2, 5.6], [4.35, 4.35], color=G, lw=1.1, ls=(0, (3, 2))); C.text(6.3, 4.35, "half the grasp force", ha="left", va="center", fontsize=7, color="#666666")
tag(C, "(c)", "Fitted grasp law, asymmetric and scaled by grasp force", dx=-0.16)

# ------------------------------------------------------------------ (d) capture and release, replayed with the law
D = fig.add_axes([0.585, 0.105, 0.40, 0.305])
for sp in ("top", "right"): D.spines[sp].set_visible(False)
D.axhline(0, color="#555555", lw=0.5)
for mode, c_, Th, lab in (("rigid", CO, 2.0, "rigid hand"), ("damp", CB, 1.0, "damping hand")):
    S = L.replay(1.0, 15.0, mode, Th, 0.25, t_after=3.2 - Th - 0.25); D.plot(S[:, 0], S[:, 1], color=c_, lw=1.5, label=lab)
    i = np.searchsorted(S[:, 0], Th); D.plot(Th, S[i, 1], "o", ms=4.5, color=c_, mec="white", mew=0.7, zorder=5)
    D.annotate("fingers open", xy=(Th, S[i, 1]), xytext=(Th, 12.5 if mode == "damp" else -15.5), ha="center", va="center", fontsize=6.5, color="#555555", arrowprops=dict(arrowstyle="-", lw=0.5, color="#888888", ls=(0, (2, 2)), shrinkA=2, shrinkB=3))
    print(mode, "spin after release %.1f deg/s, peak rebound %.1f" % (S[-1, 1], -S[:i, 1].min()))
    if mode == "rigid": D.text(3.18, S[-1, 1] + 1.4, "stored spin returns", color=c_, ha="right", va="bottom", fontsize=7)
    else: D.text(3.18, S[-1, 1] + 1.4, "spin removed", color=c_, ha="right", va="bottom", fontsize=7)
D.set_xlim(0, 3.2); D.set_ylim(-19, 19); D.set_xticks([0, 1, 2, 3]); D.set_yticks([-15, 0, 15]); D.tick_params(labelsize=7, width=0.6, length=2.5)
D.set_xlabel("time from contact (s)", fontsize=7.5); D.set_ylabel("target spin (deg/s)", fontsize=7.5); D.legend(frameon=False, fontsize=7, loc="upper right", handlelength=1.6, borderaxespad=0.1)
tag(D, "(d)", "Capture and release, replayed with the law", dx=-0.16)
fig.savefig(os.path.join(FIG, "Fig1.png")); fig.savefig(os.path.join(FIG, "Fig1.pdf"))
