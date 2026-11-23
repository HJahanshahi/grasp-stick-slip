# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""figs.py - paper Figures 4 to 8 and Table 2, from out/metrics.csv and out/signals.npz.
Paper figure <- function (file written to out/figs):
  Fig. 4 one grasp            <- fig2()             (fig2_time_histories)
  Fig. 5 closure              <- fig3()             (fig3_closure_map)
  Fig. 6 torque vs rim angle  <- fig4_angle()       (fig4_torque_angle)
  Fig. 7 hysteresis loops     <- fig5_loops()       (fig5_hysteresis_angle)
  Fig. 8 law validation       <- fig6_validation()  (fig6_law_validation)
  Table 2                     <- table2()           (out/table2_closure.md)
The other functions (fig4, fig5, fig6, fig7) draw early diagnostic plots that are not in the paper."""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); FIG = os.path.join(OUT, "figs"); os.makedirs(FIG, exist_ok=True)
C = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "pink": "#CC79A7", "grey": "#6b6b6b"}
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 600,
                     "lines.linewidth": 1.2, "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5})
rows = [r for r in csv.DictReader(open(os.path.join(OUT, "metrics.csv")))]
def f(r, k):
    try: return float(r[k])
    except Exception: return np.nan
S = np.load(os.path.join(OUT, "signals.npz"))
def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight"); plt.close(fig)
    print("saved", name)

con = [r for r in rows if r["contact"] == "1"]
# ------------------------------------------------------------------ Fig 2: one trial
def fig2(trial=None):
    cand = [r for r in con if r["block"] == "G" and r["close_speed"] == "255" and r["force"] == "100" and r["spin"] == "12"]
    r = cand[0] if trial is None else next(x for x in con if x["trial"] == trial); k = r["trial"]
    t, dF, tau, wB, dFB = S[k + "/t"], S[k + "/dF"], S[k + "/tau"], S[k + "/wB"], S[k + "/dFB"]
    grip = S[k + "/grip"]; tact = S[k + "/tact"]; t_close, t_hold, t_rel = S[k + "/events"]
    # physical events measured from the signals (the commands come earlier; latencies are given in the caption)
    p = grip[:, 1:4].mean(1); p0 = p[grip[:, 0] < t_close].mean() if (grip[:, 0] < t_close).any() else p[0]
    i = np.where((grip[:, 0] > t_close) & (np.abs(grip[:, 1:4] - grip[0, 1:4]).max(1) >= 1))[0][0]; ev_close = 0.5 * (grip[i - 1, 0] + grip[i, 0])
    w_hold = wB[(t > 0.3) & (t < t_hold)].mean()
    ev_slow = t[np.where((t > t_hold) & (wB < 0.9 * w_hold))[0][0]]
    ev_rest = t[np.where((t > t_hold) & (np.abs(wB) < 0.5))[0][0]]
    p_hold = p[(grip[:, 0] > t_hold) & (grip[:, 0] < t_rel)].mean()
    i = np.where((grip[:, 0] > t_rel) & (np.abs(grip[:, 1:4] - grip[np.where(grip[:, 0] < t_rel)[0][-1], 1:4]).max(1) >= 1))[0][0]; ev_open = 0.5 * (grip[i - 1, 0] + grip[i, 0])
    t0, t1 = t_close - 0.35, min(t[-1], t_rel + 0.6)
    fig, ax = plt.subplots(5, 1, figsize=(3.4, 7.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 0.7, 0.8, 0.8], hspace=0.42))
    leg = dict(loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=6.5, handlelength=1.8, columnspacing=1.2, borderaxespad=0.0)
    # (a) force
    ax[0].plot(t, dF, color=C["blue"], label="chaser"); ax[0].plot(t, dFB, color=C["orange"], ls="--", lw=0.9, label="target")
    ax[0].set_ylabel("force rise |ΔF| (N)"); ax[0].legend(ncol=2, **leg)
    # (b) torque with phase labels
    ax[1].plot(t, tau, color=C["blue"]); ax[1].set_ylabel(r"torque $\tau_z$ (N m)")
    i_pk = int(np.argmax(np.where((t > 0) & (t < 0.3), tau, -9)))
    ax[1].annotate("closing push", xy=(t[i_pk], tau[i_pk]), xytext=(0.30, 0.62), fontsize=6.5, ha="left", arrowprops=dict(arrowstyle="-", lw=0.5, color=C["grey"]))
    ax[1].annotate("elastic loading\n(fingers stick)", xy=(0.33, -0.33), xytext=(0.50, 0.05), fontsize=6.5, ha="left", arrowprops=dict(arrowstyle="-", lw=0.5, color=C["grey"]))
    ax[1].annotate("stick-slip", xy=(0.80, -0.75), xytext=(1.05, -0.2), fontsize=6.5, ha="left", arrowprops=dict(arrowstyle="-", lw=0.5, color=C["grey"]))
    i_sp = int(np.argmax(np.where((t > t_rel) & (t < t_rel + 0.3), tau, -9)))
    ax[1].annotate("opening\ntransient", xy=(t[i_sp], tau[i_sp]), xytext=(t[i_sp] + 0.10, -0.75), fontsize=6.5, ha="left",
                   arrowprops=dict(arrowstyle="-", lw=0.5, color=C["grey"]))
    # (c) rim rate, with the commanded spin-down ramp for reference
    ax[2].plot(t, wB, color=C["grey"], label="measured")
    ramp_t = np.array([t0, t_hold, t_hold + 0.4, t1]); ramp_w = np.array([w_hold, w_hold, 0.0, 0.0])
    ax[2].plot(ramp_t, ramp_w, color="k", lw=0.7, ls=(0, (2, 2)), label="commanded")
    ax[2].set_ylabel("rim rate\nω (deg/s)"); ax[2].set_ylim(-2, 19); ax[2].legend(ncol=2, **leg)
    # (d) finger positions
    for j, (c, ls, lab) in enumerate([(C["blue"], "-", "A"), (C["orange"], "--", "B"), (C["green"], ":", "C")]):
        ax[3].step(grip[:, 0], grip[:, 1 + j], where="post", color=c, ls=ls, lw=1.0, label=f"finger {lab}")
    ax[3].set_ylabel("finger position\n(Robotiq units)"); ax[3].legend(ncol=3, **leg)
    # (e) tactile
    m = (tact[:, 0] > t0) & (tact[:, 0] < t1); tt = tact[m]; base = tt[tt[:, 0] < ev_close, 1:].mean(0)
    for j, (c, ls) in enumerate([(C["blue"], "-"), (C["orange"], "--"), (C["green"], ":")]):
        ax[4].plot(tt[:, 0], np.linalg.norm(tt[:, 1 + 3 * j:4 + 3 * j] - base[3 * j:3 * j + 3], axis=1) / 1000, color=c, ls=ls, lw=1.0, label=f"sensor {j + 1}")
    ax[4].set_ylabel("tactile deviation\n(10³ counts)"); ax[4].legend(ncol=3, **dict(leg, bbox_to_anchor=(0.5, -0.62)))
    ax[4].set_xlabel("time from contact onset (s)")
    # events: hold band from contact to the start of opening; dashed lines at the physical events
    evs = [(ev_close, "fingers start\nclosing", "right"), (0.0, "contact\nonset", "left"),
           (ev_slow, "rim starts\nslowing", "right"), (ev_open, "fingers start\nopening", "left")]
    for i_, a in enumerate(ax):
        a.axvspan(0, ev_open, color="#000", alpha=0.045, lw=0)
        for x_, _, _ in evs: a.axvline(x_, color=C["grey"], lw=0.5, ls=(0, (3, 2)))
        a.text(0.01, 0.96, "(" + "abcde"[i_] + ")", transform=a.transAxes, fontsize=8, fontweight="bold", va="top")
    ax[2].plot([ev_rest], [0], marker="v", ms=4, color="k"); ax[2].annotate("at rest", xy=(ev_rest, 0), xytext=(ev_rest + 0.07, 2.2), fontsize=6, color="k")
    tr_ = ax[0].get_xaxis_transform()
    for x_, lab, ha in evs:
        dx = {"right": 0.02, "left": -0.02}[ha]
        ax[0].text(x_ + dx, 1.03, lab, transform=tr_, fontsize=6, color=C["grey"], ha=ha, va="bottom", linespacing=0.95)
    ax[0].set_xlim(t0, t1)
    # equalize the visible gaps: panel (b) has no legend row underneath, so move it down by half a legend row
    fig.canvas.draw()
    lg = ax[0].get_legend().get_window_extent().transformed(fig.transFigure.inverted())
    pb = ax[1].get_position(); ax[1].set_position([pb.x0, pb.y0 - 0.5 * lg.height, pb.width, pb.height])
    save(fig, "fig2_time_histories")
    print(f"Fig2 {k}: commands close {t_close:.3f} spin-down {t_hold:.3f} open {t_rel:.3f}; events fingers move {ev_close:.3f}, "
          f"rim slows {ev_slow:.3f}, rim at rest {ev_rest:.3f}, fingers open {ev_open:.3f}")

# ------------------------------------------------------------------ Fig 3: closure map vs finger speed
def closure_rows():
    return [r for r in con if r["mode"] == "kin" and r["grasp"] == "1" and r["force"] == "40" and abs(f(r, "hold") - 0.25) < 1e-6 and r["block"] in ("C", "O", "S")]

import matplotlib.cm as _cm
SPEED_COL = {cs: matplotlib.colors.to_hex(_cm.plasma(x)) for cs, x in zip((40, 100, 170, 255), (0.02, 0.32, 0.56, 0.76))}

def closure_quantities(r):
    """loading rate over 10-60 ms after onset; force against finger travel since onset (sampled at the gripper-log times,
    up to the open command); measured finger speed before contact."""
    k = r["trial"]; t, dF = S[k + "/t"], S[k + "/dF"]; g = S[k + "/grip"]; tc, te, tr = S[k + "/events"]
    a = (t >= 0.01) & (t <= 0.06); rate = np.polyfit(t[a], dF[a], 1)[0]
    gt, gp = g[:, 0], g[:, 1:4].mean(1); m = (gt >= -0.06) & (gt <= tr)
    gt_m, gp_m = gt[m], gp[m]
    # end the trace at the first stall after contact (two successive log samples with no finger motion)
    stall = np.where((gt_m[1:] > 0) & (np.abs(np.diff(gp_m)) < 0.5))[0]
    if len(stall): gt_m, gp_m = gt_m[:stall[0] + 1], gp_m[:stall[0] + 1]
    travel = gp_m - np.interp(0.0, gt, gp); Ft = np.interp(gt_m, t, dF)
    return dict(rate=rate, travel=travel, Ft=Ft, t=t, dF=dF, t_open=tr, v=abs(f(r, "finger_rate")))

def fig3():
    rr = closure_rows(); Q = {r["trial"]: closure_quantities(r) for r in rr}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.42))
    leg = dict(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=6.5, handlelength=1.4, columnspacing=1.0, borderaxespad=0.0)
    grid = np.arange(-0.02, 0.2001, 0.002)              # every closure trial is still closed at 0.20 s
    # (a) force against time since contact: mean and +-1 sd per speed (each trial cut at its open command)
    for cs in (40, 100, 170, 255):
        g = [r for r in rr if int(r["close_speed"]) == cs]
        M = np.full((len(g), len(grid)), np.nan)
        for i, r in enumerate(g):
            q = Q[r["trial"]]; ok = grid <= q["t_open"]; M[i, ok] = np.interp(grid[ok], q["t"], q["dF"])
        mu, sd = np.nanmean(M, 0), np.nanstd(M, 0); n = np.sum(np.isfinite(M), 0); keep = n == len(g)
        ax[0].plot(grid[keep], mu[keep], color=SPEED_COL[cs], lw=1.2, label=f"speed setting {cs}")
        ax[0].fill_between(grid[keep], (mu - sd)[keep], (mu + sd)[keep], color=SPEED_COL[cs], alpha=0.18, lw=0)
    ax[0].set_xlabel("time from contact (s)"); ax[0].set_ylabel("force rise |ΔF| (N)"); ax[0].set_xlim(-0.02, 0.20); ax[0].set_ylim(0, None)
    # (b) loading rate against measured finger speed
    for st, mk in (("half", "o"), ("open", "^")):
        g = [r for r in rr if r["start"] == st]
        ax[1].scatter([Q[r["trial"]]["v"] for r in g], [Q[r["trial"]]["rate"] for r in g], s=10, marker=mk, lw=0, alpha=0.75,
                      c=[SPEED_COL[int(r["close_speed"])] for r in g])
    v = np.array([Q[r["trial"]]["v"] for r in rr]); R = np.array([Q[r["trial"]]["rate"] for r in rr])
    kv = np.sum(v * R) / np.sum(v * v); xx = np.array([0, 130]); ax[1].plot(xx, kv * xx, color="k", lw=0.7)
    r2v = 1 - np.sum((R - kv * v) ** 2) / np.sum((R - R.mean()) ** 2)
    ax[1].text(6, 372, f"proportional fit\nR² = {r2v:.2f}", fontsize=6.5, va="top")
    ax[1].set_xlim(0, 130); ax[1].set_ylim(0, None)
    ax[1].set_xlabel("finger speed before contact\n(Robotiq units/s)"); ax[1].set_ylabel("loading rate (N/s)")
    ax[1].scatter([], [], marker="o", s=10, color="grey", label="half-closed start"); ax[1].scatter([], [], marker="^", s=10, color="grey", label="open start")
    # (c) force against finger travel since contact: all trials
    X, Y = [], []
    for r in rr:
        q = Q[r["trial"]]; ax[2].plot(q["travel"], q["Ft"], color=SPEED_COL[int(r["close_speed"])], lw=0.4, alpha=0.45)
        s_ = (q["travel"] >= 3) & (q["travel"] <= 12); X += list(q["travel"][s_]); Y += list(q["Ft"][s_])
    X, Y = np.array(X), np.array(Y); pk = np.polyfit(X, Y, 1)
    xx = np.array([3, 12]); ax[2].plot(xx, np.polyval(pk, xx), color="k", lw=1.0, ls=(0, (3, 2)))
    r2c = 1 - np.sum((Y - np.polyval(pk, X)) ** 2) / np.sum((Y - Y.mean()) ** 2)
    ax[2].text(-3.2, 54, f"linear fit, 3–12 units\nslope {pk[0]:.2f} N/unit\nR² = {r2c:.2f}", fontsize=6.5, va="top")
    ax[2].set_xlim(-4, 22); ax[2].set_ylim(0, 57)
    ax[2].set_xlabel("finger travel since contact\n(Robotiq units)"); ax[2].set_ylabel("force rise |ΔF| (N)")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=SPEED_COL[cs], lw=1.4, label=f"speed setting {cs}") for cs in (40, 100, 170, 255)]
    hs += [Line2D([], [], color="grey", marker="o", ls="", ms=3.5, label="half-closed start"), Line2D([], [], color="grey", marker="^", ls="", ms=3.5, label="open start")]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=6, fontsize=6.3, handlelength=1.3, columnspacing=1.0, frameon=False)
    for a_, l in zip(ax, "abc"): a_.text(-0.30, 1.04, f"({l})", transform=a_.transAxes, fontsize=8, fontweight="bold")
    save(fig, "fig3_closure_map")
    per = {cs: np.sum(np.concatenate([Q[r["trial"]]["travel"][(Q[r["trial"]]["travel"] >= 3) & (Q[r["trial"]]["travel"] <= 12)] for r in rr if int(r["close_speed"]) == cs]) *
                       np.concatenate([Q[r["trial"]]["Ft"][(Q[r["trial"]]["travel"] >= 3) & (Q[r["trial"]]["travel"] <= 12)] for r in rr if int(r["close_speed"]) == cs])) for cs in (40,)}
    print(f"Fig3: {len(rr)} trials; loading rate = {kv:.2f} N per unit x finger speed (R2 {1 - np.sum((R - kv * v) ** 2) / np.sum((R - R.mean()) ** 2):.2f}); "
          f"force-travel slope over 3-12 units {pk[0]:.2f} N/unit, intercept {pk[1]:.1f} N (R2 {1 - np.sum((Y - np.polyval(pk, X)) ** 2) / np.sum((Y - Y.mean()) ** 2):.2f})")
    return Q

# ------------------------------------------------------------------ Fig 4: tactile lag
def fig4():
    rr = [r for r in con if r["grasp"] == "1" and r["mode"] == "kin" and np.isfinite(f(r, "tact_lag_ms"))]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    bins = np.arange(-150, 400, 25)
    for st, c in [("half", C["blue"]), ("open", C["orange"])]:
        v = [f(r, "tact_lag_ms") for r in rr if r["start"] == st]
        ax.hist(v, bins=bins, color=c, alpha=0.6, label=f"start {st} (n={len(v)}, median {np.median(v):.0f} ms)")
    ax.axvline(0, color=C["grey"], ls="--", lw=0.6); ax.set_xlabel("tactile onset minus force onset (ms)"); ax.set_ylabel("trials"); ax.legend(loc="upper right", fontsize=6.5)
    save(fig, "fig4_tactile_lag")

# ------------------------------------------------------------------ Fig 5: steady torque and force vs speed
def fig5():
    rr = [r for r in con if r["mode"] == "kin" and r["grasp"] == "1" and r["block"] in ("G", "G2") and r["close_speed"] == "255"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for F_, c, mk in [("40", C["blue"], "o"), ("100", C["orange"], "s"), ("180", C["green"], "^")]:
        g = [r for r in rr if r["force"] == F_]
        sp = sorted(set(int(r["spin"]) for r in g))
        for a, key in zip(ax, ["tau_steady", "dF_steady"]):
            mu = [np.mean([f(r, key) for r in g if int(r["spin"]) == s]) for s in sp]; sd = [np.std([f(r, key) for r in g if int(r["spin"]) == s]) for s in sp]
            a.errorbar(sp, mu, yerr=sd, color=c, marker=mk, ms=4, lw=1, capsize=2, label=f"grip force setting {F_}")
            a.scatter([int(r["spin"]) for r in g], [f(r, key) for r in g], s=5, color=c, alpha=0.35, lw=0)
    ax[0].axhline(0, color=C["grey"], lw=0.6); ax[0].axvline(0, color=C["grey"], lw=0.6)
    ax[0].set_xlabel("rim rate ω (deg/s)"); ax[0].set_ylabel("steady torque τ_z (N m)"); ax[1].set_xlabel("rim rate ω (deg/s)"); ax[1].set_ylabel("steady normal force |ΔF| (N)")
    ax[0].legend(loc="upper right", fontsize=6.5)
    save(fig, "fig5_torque_vs_speed")

# ------------------------------------------------------------------ Fig 6: hysteresis loops
def fig6():
    fig, ax = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex="row", sharey="row")
    for i, w in enumerate(["12", "30"]):
        for j, F_ in enumerate(["40", "100", "180"]):
            a = ax[i, j]
            for blk, c, lab in [("H", C["blue"], "+ω first"), ("H2", C["orange"], "−ω first")]:
                g = [r for r in con if r["block"] == blk and r["force"] == F_ and abs(int(r["spin"])) == int(w) and r["close_speed"] == "255"]
                for n, r in enumerate(g):
                    k = r["trial"]; t, wB, tau = S[k + "/t"], S[k + "/wB"], S[k + "/tau"]
                    m = (t > 0.05) & (t < f(r, "hold_s"))
                    a.plot(wB[m], tau[m], color=c, lw=0.7, alpha=0.8, label=lab if n == 0 else None)
            a.axhline(0, color=C["grey"], lw=0.5); a.axvline(0, color=C["grey"], lw=0.5)
            a.set_title(f"|ω| = {w} deg/s, grip force setting {F_}", fontsize=7)
            if i == 1: a.set_xlabel("rim rate ω (deg/s)")
            if j == 0: a.set_ylabel("torque τ_z (N m)")
    ax[0, 0].legend(loc="upper right", fontsize=6.5)
    save(fig, "fig6_hysteresis_loops")

# ------------------------------------------------------------------ Fig 7: time traces of sweeps through the zero crossings (model overlay later)
def fig7():
    g = [r for r in con if r["block"] == "H" and r["force"] == "100" and r["spin"] == "30" and r["close_speed"] == "255"]
    fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.4), sharex=True)
    for n, r in enumerate(g):
        k = r["trial"]; t, wB, tau, dF = S[k + "/t"], S[k + "/wB"], S[k + "/tau"], S[k + "/dF"]
        ax[0].plot(t, wB, color=C["grey"], lw=0.8, label="rim rate" if n == 0 else None)
        ax[1].plot(t, tau, color=C["blue"], lw=0.8, alpha=0.8, label="torque (3 repeats)" if n == 0 else None)
    ax[0].set_ylabel("ω (deg/s)"); ax[1].set_ylabel("τ_z (N m)"); ax[1].set_xlabel("time from contact (s)")
    for a in ax: a.axhline(0, color=C["grey"], lw=0.5)
    ax[1].legend(loc="upper left", fontsize=6.5)
    save(fig, "fig7_sweep_traces")

# ------------------------------------------------------------------ Table 2: closure statistics
def table2():
    """closure statistics per closing-speed setting (blocks C, O, S): same quantities as Fig. 4"""
    rr = closure_rows(); Q = {r["trial"]: closure_quantities(r) for r in rr}; lines = []
    lines.append("| closing speed setting | n | finger speed before contact (units/s) | loading rate, 10–60 ms (N/s) | force–travel slope, 3–12 units (N/unit) | time to 20 N, median (IQR) (ms) | trials reaching 20 N before opening |")
    lines.append("|---|---|---|---|---|---|---|")
    for cs in (40, 100, 170, 255):
        g = [r for r in rr if int(r["close_speed"]) == cs]; q = [Q[r["trial"]] for r in g]
        v = np.array([x["v"] for x in q]); R = np.array([x["rate"] for x in q])
        X = np.concatenate([x["travel"][(x["travel"] >= 3) & (x["travel"] <= 12)] for x in q]); Y = np.concatenate([x["Ft"][(x["travel"] >= 3) & (x["travel"] <= 12)] for x in q])
        sl = np.polyfit(X, Y, 1)[0]
        # time to 20 N; trials whose fingers opened first count as "later than" (censored), so the median stays unbiased
        tt = []
        for x in q:
            m = (x["t"] >= 0) & (x["t"] <= x["t_open"]); i = np.where(x["dF"][m] >= 20)[0]
            tt.append(1000 * x["t"][m][i[0]] if len(i) else np.inf)
        tt = np.array(tt); n_r = int(np.isfinite(tt).sum())
        ts = np.sort(tt); n = len(ts)
        def pct(p):                                          # linear-interpolated percentile; "not reached" if it falls among censored trials
            h = (n - 1) * p / 100; lo, hi = int(np.floor(h)), int(np.ceil(h))
            return np.inf if not np.isfinite(ts[hi]) else ts[lo] + (h - lo) * (ts[hi] - ts[lo])
        md, q1, q3 = pct(50), pct(25), pct(75)
        fmt = lambda z: "not reached" if not np.isfinite(z) else f"{z:.0f}"
        lines.append(f"| {cs} | {len(g)} | {v.mean():.0f} ± {v.std(ddof=1):.0f} | {R.mean():.0f} ± {R.std(ddof=1):.0f} | {sl:.2f} | {fmt(md)} ({fmt(q1)} to {fmt(q3)}) | {n_r}/{len(g)} |")
    open(os.path.join(OUT, "table2_closure.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

# ------------------------------------------------------------------ Table of steady torque (for the text / Table 3 base)
def table_torque():
    rr = [r for r in con if r["mode"] == "kin" and r["grasp"] == "1" and r["block"] in ("G", "G2") and r["close_speed"] == "255"]
    sp = sorted(set(int(r["spin"]) for r in rr)); lines = ["| grip force setting | " + " | ".join(f"{s:+d}" for s in sp) + " |", "|---|" + "---|" * len(sp)]
    for F_ in ("40", "100", "180"):
        cells = []
        for s in sp:
            g = [f(r, "tau_steady") for r in rr if r["force"] == F_ and int(r["spin"]) == s]
            cells.append(f"{np.mean(g):+.2f} ± {np.std(g):.2f} (n={len(g)})" if g else "–")
        lines.append(f"| {F_} | " + " | ".join(cells) + " |")
    open(os.path.join(OUT, "table_torque.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

if __name__ == "__main__":
    fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); table2(); table_torque()

# ------------------------------------------------------------------ new Fig 4: torque against turned angle (constant-rate holds)
def fig4_angle():
    from fit_law import simulate
    P = json.load(open(os.path.join(OUT, "law_params.json")))["params"]
    rr = [r for r in con if r["mode"] == "kin" and r["grasp"] == "1" and r["block"] in ("G", "G2") and r["close_speed"] == "255"]
    rates = sorted(set(int(r["spin"]) for r in rr)); T_CL = 0.15; TH = {}
    cmap = plt.cm.PuOr
    RC = {w: ("#9a9a9a" if w == 0 else matplotlib.colors.to_hex(cmap(0.5 + 0.47 * np.sign(w) * (0.25 + 0.75 * abs(w) / 30)))) for w in rates}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.32))
    for r in rr:
        k = r["trial"]; t, w, tau = S[k + "/t"], S[k + "/wB"], S[k + "/tau"]; m = (t > 0) & (t < float(r["hold_s"]))
        th = np.cumsum(w[m] * np.r_[0, np.diff(t[m])]); cl = t[m] <= T_CL
        ax[0].plot(th, tau[m], color=RC[int(r["spin"])], lw=0.45, alpha=0.8)
        st = (t[m] > float(r["hold_s"]) - 0.5); TH[r["trial"]] = th[st].mean()
    from fit_law import smooth_F
    for w0 in (-30, 30):                                   # law driven by the measured rate and grasp force of one hold per direction
        r = next(x for x in rr if int(x["spin"]) == w0 and x["force"] == "100"); k = r["trial"]
        t, w = S[k + "/t"], S[k + "/wB"]; m = (t > 0) & (t < float(r["hold_s"]))
        th = np.cumsum(w[m] * np.r_[0, np.diff(t[m])])
        ax[0].plot(th, simulate(P, np.deg2rad(w[m]), smooth_F(S[k + "/dF"])[m]), color="k", lw=1.0, ls=(0, (3, 2)))
    ax[0].axhline(0, color=C["grey"], lw=0.5); ax[0].axvline(0, color=C["grey"], lw=0.5)
    ax[0].set_xlabel("rim angle turned since contact (deg)"); ax[0].set_ylabel("torque on the hand τ$_z$ (N m)")
    ax[0].axvspan(-33, -17.9, color="#f0f0f0", lw=0, zorder=0); ax[0].axvspan(18.8, 33, color="#f0f0f0", lw=0, zorder=0); ax[0].set_xlim(-33, 33)
    ax[0].text(-25.5, -1.75, "beyond the\nsweep range", fontsize=6, color=C["grey"], ha="center", va="bottom")
    ax[0].text(25.9, 2.2, "beyond the\nsweep range", fontsize=6, color=C["grey"], ha="center", va="bottom")
    ax[0].text(-16, 6.5, "wedging direction", fontsize=6.5, color=RC[-24]); ax[0].text(17.5, 1.6, "slip direction", fontsize=6.5, color=RC[24], ha="right")
    # (b) normal force over the last 0.5 s against rim rate
    for F_, mk in (("40", "o"), ("100", "s"), ("180", "^")):
        g = [r for r in rr if r["force"] == F_]
        ax[1].scatter([TH[r["trial"]] for r in g], [f(r, "dF_steady") for r in g], s=10, marker=mk,
                      facecolor="none", edgecolor=[RC[int(r["spin"])] for r in g], lw=0.6)
    md = [np.median([f(r, "dF_steady") for r in rr if int(r["spin"]) == w]) for w in rates]
    xa = [np.median([TH[r["trial"]] for r in rr if int(r["spin"]) == w]) for w in rates]
    ax[1].plot(xa, md, color="k", lw=0.8)
    ax[1].set_xlabel("rim angle turned since contact (deg)"); ax[1].set_ylabel("grasp force rise |ΔF| (N)"); ax[1].set_ylim(0, None)
    for i_, a in enumerate(ax): a.text(-0.14, 1.04, "(" + "ab"[i_] + ")", transform=a.transAxes, fontsize=8, fontweight="bold")
    h = [matplotlib.lines.Line2D([], [], color=RC[w], lw=1.2) for w in rates]
    hl = [matplotlib.lines.Line2D([], [], color="k", lw=1.0, ls=(0, (3, 2))), matplotlib.lines.Line2D([], [], color="k", lw=0.8)]
    hm = [matplotlib.lines.Line2D([], [], ls="", marker=m_, mfc="none", mec="k", ms=4) for m_ in "os^"]
    fig.legend(h, [f"{w:+d}".replace("-", "−") if w else "0" for w in rates], title="rim rate (deg/s)", title_fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=9, fontsize=6.5, handlelength=1.4, columnspacing=0.9)
    fig.legend(hl + hm, ["fitted law, one hold per direction", "median at each rate", "grip setting 40", "grip setting 100", "grip setting 180"],
               loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=5, fontsize=6.5, handlelength=1.8, columnspacing=1.2)
    save(fig, "fig4_torque_angle")

# ------------------------------------------------------------------ new Fig 5: hysteresis loops against turned angle, with the law
def fig5_loops():
    from fit_law import simulate
    P = json.load(open(os.path.join(OUT, "law_params.json")))["params"]
    DC = {"H": "#2c7fb8", "H2": "#d95f0e"}; LC = {"H": "#08306b", "H2": "#7f2704"}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(wspace=0.28))
    for i, amp in enumerate((12, 30)):
        a = ax[i]
        for blk in ("H", "H2"):
            g = [r for r in con if r["block"] == blk and abs(int(r["spin"])) == amp and r["close_speed"] == "255"]
            rep = next(x for x in g if x["force"] == "100")      # the law is drawn for one grip setting 100 trial per direction
            for n, r in enumerate(g):
                k = r["trial"]; t, w, tau = S[k + "/t"], S[k + "/wB"], S[k + "/tau"]; m = (t > 0) & (t < f(r, "hold_s"))
                th = np.cumsum(w[m] * np.r_[0, np.diff(t[m])])
                a.plot(th, tau[m], color=DC[blk], lw=0.5, alpha=0.6)
                if r is rep:
                    from fit_law import smooth_F
                    sim = simulate(P, np.deg2rad(w[m]), smooth_F(S[k + "/dF"])[m])
                    a.plot(th, sim, color=LC[blk], lw=1.1, ls=(0, (3, 1.5)))
                    for fr in (0.18, 0.45, 0.8):
                        j = int(fr * len(th)); j2 = j + 25
                        a.annotate("", xy=(th[j2], sim[j2]), xytext=(th[j], sim[j]), arrowprops=dict(arrowstyle="-|>", color=DC[blk], lw=0.9, mutation_scale=8))
        a.axhline(0, color=C["grey"], lw=0.5); a.axvline(0, color=C["grey"], lw=0.5)
        a.set_xlabel("rim angle turned since contact (deg)"); a.set_title(f"sweep amplitude {amp} deg/s", fontsize=7.5)
        a.text(-0.12, 1.06, "(" + "ab"[i] + ")", transform=a.transAxes, fontsize=8, fontweight="bold")
    ax[0].set_ylabel("torque on the hand τ$_z$ (N m)")
    h = [matplotlib.lines.Line2D([], [], color=DC["H"], lw=1.2), matplotlib.lines.Line2D([], [], color=DC["H2"], lw=1.2),
         matplotlib.lines.Line2D([], [], color=LC["H"], lw=1.1, ls=(0, (3, 1.5))), matplotlib.lines.Line2D([], [], color=LC["H2"], lw=1.1, ls=(0, (3, 1.5)))]
    fig.legend(h, ["measured, rim turns in the slip direction first (9 trials)", "measured, rim turns in the wedging direction first (6 trials)",
                   "fitted law, slip direction first (one trial)", "fitted law, wedging direction first (one trial)"],
               loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=6.5, handlelength=2.2, columnspacing=1.5)
    save(fig, "fig5_hysteresis_angle")

# ------------------------------------------------------------------ new Fig 6: law validation on the holds (not used in the fit)
def fig6_validation():
    V = list(csv.DictReader(open(os.path.join(OUT, "law_validation.csv"))))
    fig, a = plt.subplots(figsize=(3.4, 3.2))
    groups = [("long", lambda v: v["block"] in ("G", "G2") and v["spin"] != "0", "o", "#2c7fb8", "holds of 1 s"),
              ("short", lambda v: v["block"] in ("C", "S") and v["spin"] != "0", "^", "#d95f0e", "holds of 0.2 to 0.6 s"),
              ("rest", lambda v: v["spin"] == "0", "o", "#9a9a9a", "rim at rest (law gives zero)")]
    for name, sel, mk, c, lab in groups:
        g = [v for v in V if sel(v)]
        x = [float(v["meas"]) for v in g]; y = [float(v["pred"]) for v in g]
        if name == "rest": a.scatter(x, y, s=12, marker=mk, facecolor="none", edgecolor=c, lw=0.6, label=f"{lab}, n = {len(g)}", zorder=2)
        else: a.scatter(x, y, s=12, marker=mk, color=c, alpha=0.75, lw=0, label=f"{lab}, n = {len(g)}", zorder=3)
    mv = [v for v in V if v["spin"] != "0"]; m = np.array([float(v["meas"]) for v in mv]); p = np.array([float(v["pred"]) for v in mv]); e = p - m
    lim = [-2.5, 6.5]; a.plot(lim, lim, color="k", lw=0.6); a.set_xlim(lim); a.set_ylim(lim); a.set_aspect("equal")
    a.text(6.2, -1.6, f"rim moving, n = {len(mv)}\nRMSE {np.sqrt(np.mean(e**2)):.2f} N m\nbias {e.mean():+.2f} N m".replace("-", "−") + f"\nR² = {1 - np.sum(e**2) / np.sum((m - m.mean())**2):.2f}",
           fontsize=6.5, ha="right", va="bottom")
    a.set_xlabel("measured torque at the end of the hold (N m)"); a.set_ylabel("torque predicted by the law (N m)")
    a.legend(loc="upper left", fontsize=6.5, handletextpad=0.3, borderaxespad=0.2)
    save(fig, "fig6_law_validation")
    r = [float(v["meas"]) for v in V if v["spin"] == "0"]
    print(f"moving holds n={len(mv)} rmse {np.sqrt(np.mean(e**2)):.2f} bias {e.mean():+.2f}; rest holds measured mean {np.mean(r):.2f} (range {min(r):.2f} to {max(r):.2f})")
