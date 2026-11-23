# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""fig7_figS3_tableS4.py - Fig. 7 (capture of a six-axis free body), fig. S3 (long holds) and table S4,
from data/free_body_trials.npz. Also prints the numbers quoted in the text."""
import os, sys, csv, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "..", "figures"); os.makedirs(FIG, exist_ok=True)
DS = os.path.join(HERE, "data", "free_body_trials"); sys.argv = ["x", DS, "1"]; sys.path.insert(0, HERE)
import law_free_body as L
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Inter", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 600, "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5, "axes.axisbelow": True})
CO, CB = "#D55E00", "#0072B2"; COL = {"rigid": CO, "damp": CB}
Z = np.load(DS + ".npz", allow_pickle=True); ic = {str(c): j for j, c in enumerate(Z["em_cols"])}
rows = [r for r in csv.DictReader(open(DS + "_index.csv")) if r["use"] == "1" and r["label"] == "grasp_centred"]
T = []
for r in rows:
    E = Z[r["trial"] + "/em"]; ev = {}
    for n, tt in json.loads(str(Z[r["trial"] + "/events"])): ev.setdefault(n.split()[0], tt)
    t = E[:, 0]; g = E[:, ic["gate"]] > 0; k0 = int(np.argmax(g)); tc = t[k0]; trel = ev["release"]; w = np.rad2deg(E[:, ic["wz"]]); post = (t >= trel) & g
    Th = trel - tc; Top = L.DT; I = float(r["I_zz"]); s0 = float(r["spin"])
    tau = np.einsum("nji,nj->ni", L.rotm(E[:, ic["rx"]:ic["rz"] + 1]), E[:, ic["Tx"]:ic["Tz"] + 1])[:, 2]; m = t >= tc - 0.3
    Sim = L.replay(I, w[k0], r["hand"], Th, max(Top, L.DT))
    kg = np.flatnonzero(g & (t <= trel + 0.05)); kl = kg[-1]; tau_rel = float(np.abs(tau[kl - 12:kl + 1]).mean())
    T.append(dict(id=r["trial"], hand=r["hand"], I=I, s0=s0, Th=Th, tau_rel=tau_rel, t=t[m] - tc, w=w[m], S=Sim, M=L.metrics(t[t >= tc] - tc, w[t >= tc], tau[t >= tc], Th, s0), P=L.metrics(Sim[:, 0], Sim[:, 1], Sim[:, 3], Th, s0)))
for x in T:                      # the angle turned before the stop is defined only when the body stops while held
    for q in ("M", "P"):
        if not np.isfinite(x[q]["t_stop"]): x[q]["slip_angle"] = np.nan
    if not np.isfinite(x["M"]["t_stop"]): x["P"]["t_stop"] = np.nan; x["P"]["slip_angle"] = np.nan      # same trials for hardware and law
Is = sorted({x["I"] for x in T}); print("inertias", Is, "n", len(T))
fig = plt.figure(figsize=(7.2, 5.4)); gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.55, wspace=1.3)
for j, I in enumerate(Is[1:]):                      # 0.58, 1.16, 2.32 kg m2, start at -15 deg/s (wedging direction)
    a = fig.add_subplot(gs[0, 2 * j:2 * j + 2]); a.axhline(0, color="#6b6b6b", lw=0.5)
    for hand in ("rigid", "damp"):
        for x in [x for x in T if x["I"] == I and x["hand"] == hand and x["s0"] == -15]:
            k = x["t"] <= x["Th"] + 1.0; a.plot(x["t"][k], x["w"][k], color=COL[hand], lw=1.1)
            a.plot(x["S"][:, 0], x["S"][:, 1], color=COL[hand], lw=0.9, ls=(0, (3, 2)))
            kr = np.searchsorted(x["t"], x["Th"]); a.plot(x["t"][kr], x["w"][kr], "o", ms=3.5, color=COL[hand], mec="white", mew=0.6)
    a.set_title(f"I = {I:.2f} kg·m²", fontsize=8); a.set_xlim(-0.3, 6.3); a.set_ylim(-22, 24); a.set_xticks([0, 2, 4, 6]); a.set_xlabel("time from contact (s)")
    if j == 0: a.set_ylabel("target spin\n(deg/s)"); a.text(-0.32, 1.08, "(a)", transform=a.transAxes, fontweight="bold")
a = fig.add_subplot(gs[1, 0:2]); a.text(-0.32, 1.08, "(b)", transform=a.transAxes, fontweight="bold")
for x in T:
    if np.isfinite(x["M"]["t_stop"]) and np.isfinite(x["P"]["t_stop"]) and x["I"] > 0.3:
        a.plot(x["M"]["t_stop"], x["P"]["t_stop"], "o", ms=3.5, mec=COL[x["hand"]], mew=0.9, mfc=COL[x["hand"]] if x["s0"] > 0 else "white")
a.plot([0, 1.8], [0, 1.8], color="k", lw=0.7); a.set_xlim(0, 1.8); a.set_ylim(0, 1.8); a.set_aspect("equal"); a.set_xlabel("measured stop time (s)"); a.set_ylabel("law stop time (s)")
def bars(a, key, yl, tag):
    a.text(-0.32, 1.08, tag, transform=a.transAxes, fontweight="bold")
    for j, I in enumerate(Is):
        for hand, dx in (("rigid", -0.17), ("damp", 0.17)):
            g = [x for x in T if x["I"] == I and x["hand"] == hand]; v = np.array([abs(x["M"][key]) for x in g]); p = np.array([abs(x["P"][key]) for x in g])
            for x, vv in zip(g, v): a.plot(j + dx + np.random.default_rng(int(x["id"])).uniform(-0.07, 0.07), vv, "o", ms=3, mec=COL[hand], mew=0.8, mfc=COL[hand] if x["s0"] > 0 else "white")
            a.plot([j + dx - 0.13, j + dx + 0.13], [np.nanmean(v)] * 2, color=COL[hand], lw=2.2); a.plot([j + dx - 0.13, j + dx + 0.13], [np.nanmean(p)] * 2, color=COL[hand], lw=1.3, ls=(0, (2, 1.5)))
            print(key, I, hand, len(g), "meas %.2f law %.2f" % (np.nanmean(v), np.nanmean(p)))
    a.set_xticks(range(len(Is))); a.set_xticklabels([f"{I:.2f}" for I in Is]); a.set_xlabel("target inertia I (kg·m²)"); a.set_ylabel(yl); a.set_ylim(0, None); a.grid(axis="x", visible=False)
ac = fig.add_subplot(gs[1, 2:4]); bars(ac, "rebound", "rebound while held (deg/s)", "(c)"); ac.set_ylim(0, 26); ad = fig.add_subplot(gs[1, 4:6]); bars(ad, "w_end", "spin at release (deg/s)", "(d)"); ad.set_ylim(0, 11.5)
ac.axvspan(-0.5, 0.5, color="#000000", alpha=0.07, lw=0); ad.axvspan(-0.5, 2.5, color="#000000", alpha=0.07, lw=0); ac.set_xlim(-0.5, 3.5); ad.set_xlim(-0.5, 3.5)
ac.text(0.0, 0.985, "emulator\ndamping", transform=ac.get_xaxis_transform(), ha="center", va="top", fontsize=6, color="#555555")
ad.text(1.0, 0.985, "emulator damping", transform=ad.get_xaxis_transform(), ha="center", va="top", fontsize=6, color="#555555")
from matplotlib.lines import Line2D
h = [Line2D([], [], color=CO, lw=1.2, label="rigid hand, hardware"), Line2D([], [], color=CO, lw=1, ls=(0, (3, 2)), label="rigid hand, law replay"),
     Line2D([], [], color=CB, lw=1.2, label="damping hand, hardware"), Line2D([], [], color=CB, lw=1, ls=(0, (3, 2)), label="damping hand, law replay"),
     Line2D([], [], marker="o", ls="", ms=3.5, color="k", label="positive starting spin"), Line2D([], [], marker="o", ls="", ms=3.5, mfc="white", color="k", label="negative starting spin")]
fig.legend(handles=h, loc="upper center", ncol=3, fontsize=7); fig.subplots_adjust(top=0.86, bottom=0.09, left=0.09, right=0.99)
fig.savefig(os.path.join(FIG, "Fig7.png")); fig.savefig(os.path.join(FIG, "Fig7.pdf"))
import collections; print(collections.Counter((x["I"], x["hand"], x["s0"]) for x in T))

# ---------------- overall numbers for the text
def A(k, src, g=None):
    g = T if g is None else g
    return np.array([x[src][k] for x in g], float)
T68 = T; T = [x for x in T68 if x["I"] > 0.3]; print("\nOVERALL (I >= 0.58) n", len(T))
for k, u in (("t_stop", "s"), ("slip_angle", "deg"), ("tau_slip", "N m"), ("rebound", "deg/s")):
    a, b = A(k, "M"), A(k, "P"); ok = np.isfinite(a) & np.isfinite(b); e = b[ok] - a[ok]
    print(f"{k:10s} n={ok.sum()} measured mean {np.mean(np.abs(a[ok])):.2f}  RMS err {np.sqrt(np.mean(e**2)):.2f} {u}  bias {e.mean():+.2f}  r {np.corrcoef(a[ok], b[ok])[0,1]:.2f}  R2 {1-np.sum(e**2)/np.sum((a[ok]-a[ok].mean())**2):.2f}")
for hand in ("rigid", "damp"):
    g = [x for x in T if x["hand"] == hand]; print(hand, len(g), "rebound meas %.1f +- %.1f law %.1f | residual meas %.2f law %.2f | hold %.1f-%.1f" % (np.nanmean(A("rebound","M",g)), np.nanstd(A("rebound","M",g),ddof=1), np.nanmean(A("rebound","P",g)), np.mean(np.abs(A("w_end","M",g))), np.mean(np.abs(A("w_end","P",g))), min(x["Th"] for x in g), max(x["Th"] for x in g)))
    for sg in (-1, 1):
        gg = [x for x in g if np.sign(x["s0"]) == sg]; print("   sign", sg, len(gg), "stop %.2f/%.2f  angle %.1f/%.1f  torque %.2f/%.2f rebound %.1f/%.1f" % tuple(np.nanmean(A(k, s, gg)) for k in ("t_stop","slip_angle","tau_slip","rebound") for s in ("M","P")))
T = T68
# ---------------- Table S4
ms = lambda v: f"{np.nanmean(v):.2f} ± {np.nanstd(v, ddof=1):.2f}"; ms1 = lambda v: f"{np.nanmean(v):.1f} ± {np.nanstd(v, ddof=1):.1f}"
with open(os.path.join(FIG, "TableS4.csv"), "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["I (kg·m²)", "Hand", "n", "Starting spin (deg/s)", "Contact to release (s)", "Stop time, hardware (s)", "Stop time, law (s)", "Slip angle, hardware (deg)", "Slip angle, law (deg)", "Rebound while held, hardware (deg/s)", "Rebound while held, law (deg/s)", "Spin at release, hardware (deg/s)", "Spin at release, law (deg/s)", "Torque at release, hardware (N·m)"])
    for I in Is:
        for hand in ("rigid", "damp"):
            g = [x for x in T if x["I"] == I and x["hand"] == hand]; sp = sorted({abs(x["s0"]) for x in g})
            row = [f"{I:.2f}", "rigid" if hand == "rigid" else "damping", len(g), "±" + ", ±".join(f"{s:.0f}" for s in sp), f"{np.mean([x['Th'] for x in g]):.1f}",
                   ms(A("t_stop","M",g)), ms(A("t_stop","P",g)), ms1(A("slip_angle","M",g)), ms1(A("slip_angle","P",g)), ms1(A("rebound","M",g)), ms1(A("rebound","P",g)), ms1(np.abs(A("w_end","M",g))), ms1(np.abs(A("w_end","P",g))), ms(np.array([x["tau_rel"] for x in g]))]
            wr.writerow(row); print(" | ".join(map(str, row)))
            print("      nan stop meas/law:", int(np.isnan(A("t_stop","M",g)).sum()), int(np.isnan(A("t_stop","P",g)).sum()))
# ---------------- Fig S3: 8 s holds
rd = [r for r in csv.DictReader(open(DS + "_index.csv")) if r["use"] == "1" and r["label"] == "grasp_ringdown"]
fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.0), sharex=True, sharey=True)
for r in rd:
    E = Z[r["trial"] + "/em"]; ev = {}
    for n, tt in json.loads(str(Z[r["trial"] + "/events"])): ev.setdefault(n.split()[0], tt)
    t = E[:, 0]; g = E[:, ic["gate"]] > 0; k0 = int(np.argmax(g)); tc = t[k0]; trel = ev["release"]; w = np.rad2deg(E[:, ic["wz"]]); post = (t >= trel) & g
    Th = trel - tc; Top = L.DT; I = float(r["I_zz"]); s0 = float(r["spin"])
    Sim = L.replay(I, w[k0], "rigid", Th, max(Top, L.DT)); a = ax[0 if s0 < 0 else 1, 0 if I < 2 else 1]; m = (t >= tc - 0.3) & (t <= trel + 1.0)
    a.axhline(0, color="#6b6b6b", lw=0.5); a.plot(t[m] - tc, w[m], color=CO, lw=1.1); a.plot(Sim[:, 0], Sim[:, 1], color=CO, lw=0.8, ls=(0, (3, 2)))
    a.plot(Th, w[np.searchsorted(t, trel)], "o", ms=3.5, color=CO, mec="white", mew=0.6)
    Mm = L.metrics(t[t >= tc] - tc, w[t >= tc], None, Th, s0); Pm = L.metrics(Sim[:, 0], Sim[:, 1], None, Th, s0)
    print("ring", r["trial"], I, s0, "half period %.2f/%.2f swing1 %.1f/%.1f decay %.2f/%.2f wend %.1f/%.1f" % (Mm["half_period"], Pm["half_period"], Mm["swing1"], Pm["swing1"], Mm["decay"], Pm["decay"], abs(Mm["w_end"]), abs(Pm["w_end"])))
for j, I in enumerate((1.16, 2.32)): ax[0, j].set_title(f"I = {I:.2f} kg·m²", fontsize=8); ax[1, j].set_xlabel("time from contact (s)")
for i, lab in enumerate(("(a)", "(b)")): ax[i, 0].set_ylabel("target spin\n(deg/s)"); ax[i, 0].text(-0.2, 1.05, lab, transform=ax[i, 0].transAxes, fontweight="bold")
ax[0, 0].set_xlim(-0.3, 9.6); ax[0, 0].set_ylim(-30, 34); ax[0, 0].set_xticks([0, 2, 4, 6, 8]); ax[0, 0].set_yticks([-20, 0, 20])
for i, lab in enumerate(("start at −25 deg/s", "start at +25 deg/s")):
    for j in range(2): ax[i, j].text(0.985, 0.95, lab, transform=ax[i, j].transAxes, ha="right", va="top", fontsize=7, color="#333333")
    ax[i, 0].set_facecolor("#ededed"); ax[i, 0].grid(color="#ffffff"); ax[i, 0].text(0.985, 0.05, "emulator damping", transform=ax[i, 0].transAxes, ha="right", va="bottom", fontsize=6, color="#555555")
fig.legend(handles=h[:2] + [Line2D([], [], marker="o", ls="", ms=3.5, color=CO, label="fingers open")], loc="upper center", ncol=3, fontsize=7); fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(FIG, "FigS3.png")); fig.savefig(os.path.join(FIG, "FigS3.pdf"))
