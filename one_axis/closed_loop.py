# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""closed_loop.py - session 3: rigid hand vs damping hand on the emulated free spin (trials6, v13 runner)."""
import os, json, glob, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "out", "figs")
C = {"blue": "#0072B2", "orange": "#D55E00", "grey": "#6b6b6b"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 600,
                     "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5})
K = 24.3
HAND_SIGN_PLOT = -1.0     # logged hand angle is in the chaser wrist frame (measured hand_sign = -1); replay is in the law frame

def load(folder):
    out = []
    for d in sorted(glob.glob(os.path.join(HERE, "data3", "raw", folder, "trial_*"))):
        mp = os.path.join(d, "meta.json")
        if not os.path.exists(mp): continue
        m = json.load(open(mp)); r = m["row"]
        if r.get("mode") not in ("rigid", "damp") or "free_log" not in m: continue
        L = np.array(m["free_log"], float)
        ev = {n.split()[0]: t for n, t in m["events"]}
        t_on = next(t for n, t in m["events"] if n.startswith("free_spin_on")); t_rel = ev["release"]
        w_end = float(re.search(r"w_end=(-?[\d.]+)", " ".join(n for n, _ in m["events"])).group(1))
        held = L[:, 0] < t_rel
        post = L[:, 0] >= t_rel; gate = post & (L[:, 5] > 0)
        J = float(np.trapezoid(L[gate, 2], L[gate, 0])) if gate.sum() > 1 else 0.0
        out.append(dict(J=J, gate_end=float(L[gate, 0].max() - t_on) if gate.any() else t_rel - t_on, dry=(r["split"] == "dry"), trial_id=r["trial"], mode=r["mode"], I=float(r["inertia"]), w0=float(r["spin"]), split=r["split"],
                        trial=r["trial"], t=L[:, 0] - t_on, t_rel=t_rel - t_on, w=np.rad2deg(L[:, 3]), tau=L[:, 2],
                        hand=np.rad2deg(L[:, 6]) if L.shape[1] > 6 else np.zeros(len(L)),
                        w_rel=float(np.rad2deg(L[held, 3][-1])), w_end=w_end, peak=float(np.rad2deg(np.abs(L[held, 3]).max())),
                        tau_rel=float(L[held, 2][-1]), E0=0.5 * float(r["inertia"]) * np.deg2rad(float(r["spin"])) ** 2))
    return out

T = load("contact_data")
print(f"{len(T)} trials (v13, trials6)")
print(" I    policy  n   |w_end| mean (range)        peak held   torque at release   stored-energy prediction |tau|/sqrt(KI)")
for I in (0.5, 1.0, 2.0):
    for p in ("rigid", "damp"):
        g = [x for x in T if x["I"] == I and x["mode"] == p and x["trial"] not in ("P1", "P2")]
        we = np.abs([x["w_end"] for x in g]); pk = [x["peak"] for x in g]; tr = np.abs([x["tau_rel"] for x in g])
        pred = np.rad2deg(tr / np.sqrt(K * I))
        print(f" {I:<4} {p:<7} {len(g)}   {we.mean():5.1f} ({we.min():.1f}-{we.max():.1f})   {np.mean(pk):6.1f}     {tr.mean():.2f} N m          {pred.mean():5.1f} deg/s")
# residual vs prediction from the stored torque, all trials
x = np.array([np.rad2deg(abs(t["tau_rel"]) / np.sqrt(K * t["I"])) for t in T]); y = np.array([abs(t["w_end"] - t["w_rel"]) for t in T])
print(f"spin added by the release vs |tau_rel|/sqrt(KI): corr {np.corrcoef(x, y)[0,1]:.2f}, slope {np.polyfit(x, y, 1)[0]:.2f}")

# figure
T = [x for x in T if not x["dry"]]
import fig7_replay as RP                                   # law replay of every trial (force-scaled law + grasp-force model)
RPT = {x["trial"]: x for x in RP.trials()}
for x in T:
    y = RPT[x["trial"]]; s = RP.replay(y); x["rp_t"] = s[:, 0] + y["t_c"]; x["rp_w"] = s[:, 1]; x["rp_tau"] = s[:, 3]; x["rp_hand"] = s[:, 2]; x["rp_end"] = s[-1, 1]
fig = plt.figure(figsize=(7.2, 7.6))
gs = fig.add_gridspec(5, 6, height_ratios=[1.2, 1.0, 0.7, 0.42, 1.3], hspace=0.22, wspace=1.2)
axw = [fig.add_subplot(gs[0, 2 * j:2 * j + 2]) for j in range(3)]
axt = [fig.add_subplot(gs[1, 2 * j:2 * j + 2], sharex=axw[j]) for j in range(3)]
axh = [fig.add_subplot(gs[2, 2 * j:2 * j + 2], sharex=axw[j]) for j in range(3)]
axs = fig.add_subplot(gs[4, 0:3]); axv = fig.add_subplot(gs[4, 3:6])
for j, I in enumerate((0.5, 1.0, 2.0)):
    for p, c in (("rigid", C["orange"]), ("damp", C["blue"])):
        g = [x for x in T if x["I"] == I and x["mode"] == p and x["w0"] == 15.0]
        for n, x in enumerate(g):
            m = x["t"] < x["t_rel"] + 1.2
            lab = ("rigid hand, hardware" if p == "rigid" else "damping hand, hardware") if n == 0 else None
            axw[j].plot(x["t"][m], x["w"][m], color=c, lw=0.9, label=lab)
            i_r = np.searchsorted(x["t"], x["t_rel"])
            axw[j].plot(x["t"][i_r], x["w"][i_r], "o", ms=3.5, color=c, mec="white", mew=0.6, zorder=5)
            mt = x["t"] <= x["gate_end"]                    # contact torque only while the fingers are on the rim
            axt[j].plot(x["t"][mt], x["tau"][mt], color=c, lw=0.8)
            if p == "damp": axh[j].plot(x["t"][x["t"] <= x["t_rel"]], x["hand"][x["t"] <= x["t_rel"]], color=c, lw=0.9)
            mr = x["rp_t"] <= x["t_rel"] + 1.2; DS = (0, (3, 1.5))
            axw[j].plot(x["rp_t"][mr], x["rp_w"][mr], color=c, lw=0.8, ls=DS, alpha=0.85, label=(("rigid hand, law replay" if p == "rigid" else "damping hand, law replay") if n == 0 else None))
            axt[j].plot(x["rp_t"][mr], x["rp_tau"][mr], color=c, lw=0.8, ls=DS, alpha=0.85)
            if p == "damp": axh[j].plot(x["rp_t"][x["rp_t"] <= x["t_rel"]], HAND_SIGN_PLOT * x["rp_hand"][x["rp_t"] <= x["t_rel"]], color=c, lw=0.8, ls=DS, alpha=0.85)
    axw[j].set_title(f"I = {I} kg m²", fontsize=8, pad=4)
    for a in (axw[j], axt[j], axh[j]): a.axhline(0, color=C["grey"], lw=0.5)
    axh[j].set_xlabel("time from contact (s)"); axw[j].set_xlim(0, 3.6)
    for a in (axw[j], axt[j]): a.tick_params(labelbottom=False)
from matplotlib.ticker import MultipleLocator
for a in axh: a.yaxis.set_major_locator(MultipleLocator(2.5)); a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212")))
for row in (axw, axt, axh):
    lo = min(a.get_ylim()[0] for a in row); hi = max(a.get_ylim()[1] for a in row)
    for a in row: a.set_ylim(lo, hi)
axw[0].set_ylabel("target spin\n(deg/s)"); axt[0].set_ylabel("torque on the\ntarget (N m)"); axh[0].set_ylabel("hand angle\n(deg)")
hh, ll = axw[0].get_legend_handles_labels()
fig.legend(hh, ll, loc="lower center", bbox_to_anchor=(0.5, 0.905), ncol=4, fontsize=6.5, handlelength=2.0, columnspacing=1.4)
for a, l in zip([axw[0], axt[0], axh[0]], "abc"): a.text(-0.48, 1.04, f"({l})", transform=a.transAxes, fontsize=8, fontweight="bold")
# (d) residual spin after release, every trial
rng = np.random.default_rng(1)
for p, c, dx in (("rigid", C["orange"], -0.06), ("damp", C["blue"], 0.06)):
    for I in (0.5, 1.0, 2.0):
        g = [x for x in T if x["I"] == I and x["mode"] == p]
        for x in g:
            xx = np.log2(I) + dx + rng.uniform(-0.025, 0.025)
            axs.plot(xx, abs(x["w_end"]), "o", ms=4, color=c, mfc=c if x["w0"] > 0 else "white", mew=0.9)
        axs.plot([np.log2(I) + dx - 0.06, np.log2(I) + dx + 0.06], [np.mean([abs(x["w_end"]) for x in g])] * 2, color=c, lw=2)
        axs.plot([np.log2(I) + dx - 0.06, np.log2(I) + dx + 0.06], [np.mean([abs(x["rp_end"]) for x in g])] * 2, color=c, lw=1.2, ls=(0, (2, 1)))
axs.set_xticks([-1, 0, 1]); axs.set_xticklabels(["0.5", "1", "2"]); axs.set_xlabel("target inertia I (kg m²)"); axs.set_ylabel("spin after release (deg/s)")
axs.plot([], [], "o", color=C["orange"], label="rigid hand"); axs.plot([], [], "o", color=C["blue"], label="damping hand")
axs.plot([], [], color="k", lw=2, label="mean, hardware"); axs.plot([], [], color="k", lw=1.2, ls=(0, (2, 1)), label="mean, law replay")
axs.plot([], [], "o", color="k", mfc="k", ms=4, label="ω₀ = +15 deg/s"); axs.plot([], [], "o", color="k", mfc="white", ms=4, label="ω₀ = −15 deg/s")
axs.legend(fontsize=6, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.6); axs.set_ylim(0, 25)
axs.text(-0.12, 1.07, "(d)", transform=axs.transAxes, fontsize=8, fontweight="bold")
# (e) spin change at release against release impulse / I
for p, c in (("rigid", C["orange"]), ("damp", C["blue"])):
    g = [x for x in T if x["mode"] == p]
    axv.scatter([np.rad2deg(x["J"] / x["I"]) for x in g], [x["w_end"] - x["w_rel"] for x in g], s=14, color=c, lw=0)
lim = [-12, 12]; axv.plot(lim, lim, color="k", lw=0.6); axv.set_xlim(lim); axv.set_ylim(lim)
axv.set_xlabel("release impulse / I (deg/s)"); axv.set_ylabel("spin change at release (deg/s)")
axv.text(-0.12, 1.07, "(e)", transform=axv.transAxes, fontsize=8, fontweight="bold")
fig.savefig(os.path.join(FIG, "fig7_closed_loop.png"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, "fig7_closed_loop.pdf"), bbox_inches="tight")
e = np.array([x["w_end"] - x["w_rel"] - np.rad2deg(x["J"] / x["I"]) for x in T]); print(f"(e) RMS difference {np.sqrt(np.mean(e**2)):.2f} deg/s over {len(T)} trials")

# ---------------- Table 4: closed-loop release, per inertia and policy (24 trials)
L = ["| I (kg m²) | hand | n | spin after release, hardware, mean ± sd (deg/s) | spin after release, law replay, mean ± sd (deg/s) | |torque| at release (N m) | peak spin while held (deg/s) | contact to release (s) | peak hand turn (deg) |",
     "|---|---|---|---|---|---|---|---|---|"]
for I in (0.5, 1.0, 2.0):
    for p, name in (("rigid", "rigid"), ("damp", "damping")):
        g = [x for x in T if x["I"] == I and x["mode"] == p]
        we = np.abs([x["w_end"] for x in g]); tr = np.abs([x["tau_rel"] for x in g]); pk = [x["peak"] for x in g]
        hl = [x["t_rel"] for x in g]; hd = [np.abs(x["hand"][x["t"] <= x["t_rel"]]).max() for x in g]
        rp = np.abs([x["rp_end"] for x in g])
        L.append(f"| {I} | {name} | {len(g)} | {we.mean():.1f} ± {we.std(ddof=1):.1f} | {rp.mean():.1f} ± {rp.std(ddof=1):.1f} | {tr.mean():.2f} | {np.mean(pk):.1f} | {np.mean(hl):.2f} | {np.mean(hd):.1f} |")
r = np.array([abs(x["w_end"]) for x in T if x["mode"] == "rigid" and x["I"] >= 1]); d = np.array([abs(x["w_end"]) for x in T if x["mode"] == "damp" and x["I"] >= 1])
rr = np.array([abs(x["rp_end"]) for x in T if x["mode"] == "rigid" and x["I"] >= 1]); dd = np.array([abs(x["rp_end"]) for x in T if x["mode"] == "damp" and x["I"] >= 1])
L.append(f"| 1 and 2 | rigid / damping | 8 / 8 | {r.mean():.1f} / {d.mean():.1f} (ratio {r.mean()/d.mean():.1f}) | {rr.mean():.1f} / {dd.mean():.1f} (ratio {rr.mean()/dd.mean():.1f}) | | | | |")
open(os.path.join(HERE, "out", "table4_closed_loop.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
