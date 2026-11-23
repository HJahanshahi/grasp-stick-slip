# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""
fit_law.py - fits the force-scaled asymmetric stick-slip (Maxwell-slip / Prandtl-Ishlinskii) law to the 48 rate
sweeps of blocks H and H2 and tests it on the 220 constant-rate holds of blocks C, S, G and G2.

Law (Section III-B of the paper): N_EL = 3 elasto-slip elements in parallel, s = F / F_REF (F_REF = 50 N),
    e_j[n+1] = clip(e_j[n] - s k_j w[n] dt, -s c_j+, s c_j-),     tau[n] = sum_j e_j[n],   dt = 1/512 s
w is the rim rate (rad/s, positive in the slip direction), F the grasp force (smoothed force rise, 50 ms).
Fit: bounded least squares with a soft-L1 loss (scale 0.5 N m), 100 bootstrap resamples of whole sweeps,
separate fits per grip-force setting.
Run:  python3 fit_law.py
Outputs: out/law_params.json, out/law_bootstrap.npy, out/law_validation.csv, out/table3_law.md
         (plus two diagnostic figures, out/figs/fig7_model_overlay.* and out/figs/fig8_law_validation.*,
          which are not the paper figures; the paper figures come from figs.py)
"""
import os, csv, json
import numpy as np
from scipy.optimize import least_squares
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); FIG = os.path.join(OUT, "figs")
rows = [r for r in csv.DictReader(open(os.path.join(OUT, "metrics.csv"))) if r["contact"] == "1"]
S = np.load(os.path.join(OUT, "signals.npz"))
C = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "grey": "#6b6b6b"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 600,
                     "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5})
DT = 1.0 / 512                      # recorder sample time (the old fit assumed 2 ms)
F_REF = 50.0                         # reference grasp force (N): element parameters are quoted at this force

N_EL = 3
def smooth_F(dF, n=25):
    """grasp force for the law: 50 ms moving average of the measured force rise (removes chatter)"""
    return np.convolve(dF, np.ones(n) / n, "same")

def simulate(p, w, F=None, dt=DT):
    """force-scaled asymmetric Prandtl-Ishlinskii (Iwan) stick-slip law with N_EL parallel elasto-slip elements.
    Element i has stiffness k_i, slips at -c_i^+ when the rim moves in the positive direction and at +c_i^-
    in the wedging (negative) direction; all three scale with the grasp force F (Coulomb contact):
    s = F / F_REF. p = (k_1..k_n, c_1^+..c_n^+, c_1^-..c_n^-) at F = F_REF. F=None means F = F_REF."""
    k = np.asarray(p[:N_EL]); cp = np.asarray(p[N_EL:2 * N_EL]); cn = np.asarray(p[2 * N_EL:3 * N_EL])
    sc = np.ones(len(w)) if F is None else np.maximum(np.broadcast_to(np.asarray(F, float), (len(w),)), 5.0) / F_REF
    e = np.zeros(N_EL); tau = np.empty(len(w))
    for i, wi in enumerate(w):
        s_ = sc[i]; e = np.clip(e - s_ * k * wi * dt, -s_ * cp, s_ * cn)
        tau[i] = e.sum()
    return tau

def sweep_trials():
    return [r for r in rows if r["block"] in ("H", "H2")]

_CACHE = {}
def _stack(trials):
    """pad the sweep rate/torque signals of a trial list into (T, n) arrays, decimated to 100 Hz"""
    key = tuple(r["trial"] for r in trials)
    if key in _CACHE: return _CACHE[key]
    ws, ts, ms, fs = [], [], [], []
    for r in trials:
        k = r["trial"]; t, wB, tau = S[k + "/t"], np.deg2rad(S[k + "/wB"]), S[k + "/tau"]
        m = (t > 0.0) & (t < float(r["hold_s"]))
        ws.append(wB[m][::5]); ts.append(tau[m][::5]); ms.append((t[m][::5] > 0.1)); fs.append(smooth_F(S[k + "/dF"])[m][::5])
    T = max(len(x) for x in ws); n = len(ws)
    W = np.zeros((T, n)); Y = np.zeros((T, n)); M = np.zeros((T, n), bool); Fm = np.full((T, n), F_REF)
    for j in range(n):
        W[:len(ws[j]), j] = ws[j]; Y[:len(ts[j]), j] = ts[j]; M[:len(ms[j]), j] = ms[j]; Fm[:len(fs[j]), j] = fs[j]
    _CACHE[key] = (W, Y, M, Fm); return _CACHE[key]

def simulate_batch(p, W, Fm, dt=5 * DT):
    k = np.asarray(p[:N_EL]); cp = np.asarray(p[N_EL:2 * N_EL]); cn = np.asarray(p[2 * N_EL:3 * N_EL])
    e = np.zeros((W.shape[1], N_EL)); out = np.empty_like(W); SC = np.maximum(Fm, 5.0) / F_REF
    for i in range(W.shape[0]):
        s_ = SC[i][:, None]
        e = np.clip(e - s_ * k * W[i][:, None] * dt, -s_ * cp, s_ * cn); out[i] = e.sum(1)
    return out

def residuals(p, trials, smooth=False):
    W, Y, M, Fm = _stack(trials)
    return (simulate_batch(p, W, Fm) - Y)[M]

def fit(trials, smooth=False):
    p0 = [20.0, 8.0, 3.0,   0.3, 0.3, 0.3,   0.5, 3.0, 40.0]
    lb = [0.0] * 9; ub = [400.0] * 3 + [5.0] * 3 + [60.0] * 3
    sol = least_squares(residuals, p0, args=(trials, smooth), bounds=(lb, ub), loss="soft_l1", f_scale=0.5)
    return sol.x, float(np.sqrt(np.mean(sol.fun ** 2)))

def main():
    tr = sweep_trials()
    p, rmse = fit(tr)
    print("hardening law:", np.round(p, 3), "rmse", round(rmse, 3))
    # bootstrap over trials (piecewise)
    rng = np.random.default_rng(0); B = []
    for b in range(100):
        idx = rng.integers(0, len(tr), len(tr)); B.append(fit([tr[i] for i in idx])[0])
    B = np.array(B); lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    # per-force fits (to show the grip-force setting does not matter)
    perF = {F_: fit([r for r in tr if r["force"] == F_])[0] for F_ in ("40", "100", "180")}
    # validation on the holds: predict the end-of-hold torque
    val = []
    for r in rows:
        if r["block"] not in ("G", "G2", "C", "S") or r["grasp"] != "1" or r["mode"] != "kin": continue
        k = r["trial"]
        if k + "/t" not in S: continue
        t, wB, tau = S[k + "/t"], np.deg2rad(S[k + "/wB"]), S[k + "/tau"]
        m = (t > 0.0) & (t < float(r["hold_s"]))
        sim = simulate(p, wB[m], smooth_F(S[k + "/dF"])[m]); st = t[m] > float(r["hold_s"]) - min(0.5, max(0.1, float(r["hold_s"]) - 0.2))
        val.append(dict(trial=k, block=r["block"], spin=r["spin"], force=r["force"], hold=r["hold_s"], meas=float(tau[m][st].mean()), pred=float(sim[st].mean())))
    with open(os.path.join(OUT, "law_validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(val[0].keys())); w.writeheader(); w.writerows(val)
    e = np.array([v["pred"] - v["meas"] for v in val]); mv = np.array([v["meas"] for v in val])
    print(f"validation on {len(val)} holds: RMSE {np.sqrt(np.mean(e**2)):.2f} N m, bias {e.mean():+.2f}, R^2 {1 - np.sum(e**2)/np.sum((mv-mv.mean())**2):.3f}")
    for blk in ("G", "G2", "C", "S"):
        ee = np.array([v["pred"] - v["meas"] for v in val if v["block"] == blk])
        if len(ee): print(f"   {blk}: n={len(ee)} RMSE {np.sqrt(np.mean(ee**2)):.2f}")
    json.dump(dict(params=p.tolist(), ci_lo=lo.tolist(), ci_hi=hi.tolist(), rmse_fit=rmse, n_elements=N_EL,
                   per_force={k_: v.tolist() for k_, v in perF.items()},
                   validation=dict(n=len(val), rmse=float(np.sqrt(np.mean(e**2))), bias=float(e.mean()))), open(os.path.join(OUT, "law_params.json"), "w"), indent=1)
    # Table 3: identifiable summaries of the law (elements are exchangeable, so report derived quantities)
    def derived(q):
        """identifiable quantities from first loading at the reference force, one direction at a time (the elements
        themselves are exchangeable): initial stiffness (first 1 deg), stiffness after slip or yield and the intercept of
        that line (12-17 deg, inside the 18 deg reached in the sweeps)"""
        out = []
        for sg in (-1, 1):                                   # -1: wedging direction, +1: slip direction
            w = np.full(int(1.0 / DT), sg * np.deg2rad(20)); tau = simulate(q, w); th = np.abs(np.cumsum(w) * DT)
            m0 = th < np.deg2rad(1); m1 = (th > np.deg2rad(12)) & (th < np.deg2rad(17))
            k0 = abs(np.polyfit(th[m0], tau[m0], 1)[0]); k1, b1 = np.polyfit(th[m1], tau[m1], 1)
            out += [k0, abs(b1), abs(k1)]
        return np.array(out)
    D = derived(p); DB = np.array([derived(b) for b in B]); dlo, dhi = np.percentile(DB, [2.5, 97.5], axis=0)
    DF = {k_: derived(v) for k_, v in perF.items()}
    names = ["wedging direction: stiffness before yield (N m/rad)", "wedging direction: yield torque (N m)",
             "wedging direction: hardening stiffness after yield (N m/rad)", "slip direction: stiffness before slip (N m/rad) †",
             "slip direction: slip torque (N m)", "slip direction: stiffness after slip (N m/rad)"]
    L = [f"All quantities at a grasp force of {F_REF:.0f} N; they scale in proportion to the grasp force.", "",
         "| quantity | all sweeps (95 % CI) | grip setting 40 | grip setting 100 | grip setting 180 |", "|---|---|---|---|---|"]
    for i, n in enumerate(names):
        L.append(f"| {n} | {D[i]:.2f} ({dlo[i]:.2f}, {dhi[i]:.2f}) | {DF['40'][i]:.2f} | {DF['100'][i]:.2f} | {DF['180'][i]:.2f} |")
    L.append(f"| fit RMSE on 48 sweeps (N m) | {rmse:.2f} | | | |"); mv = np.array([v["spin"] != "0" for v in val]); em = e[mv]; mm = np.array([v["meas"] for v in val])[mv]
    L.append(f"| validation on {mv.sum()} holds with the rim moving: RMSE, bias (N m); R² | {np.sqrt(np.mean(em**2)):.2f}, {em.mean():+.2f}; {1 - np.sum(em**2) / np.sum((mm - mm.mean())**2):.2f} | | | |")
    L += ["", "† Not resolved by these data: over the first 3 deg in the slip direction the measured torque is dominated by the closing push, which the law does not model."]
    open(os.path.join(OUT, "table3_law.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
    np.save(os.path.join(OUT, "law_bootstrap.npy"), B)
    # Fig 7: sweep traces with the model
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.0), sharex="col")
    for j, (blk, w0) in enumerate([("H", "30"), ("H2", "-12")]):
        g = [r for r in rows if r["block"] == blk and r["force"] == "100" and r["spin"] == w0 and r["close_speed"] == "255"]
        for n, r in enumerate(g):
            k = r["trial"]; t, wB, tau = S[k + "/t"], S[k + "/wB"], S[k + "/tau"]; m = (t > 0) & (t < float(r["hold_s"]))
            ax[0, j].plot(t[m], wB[m], color=C["grey"], lw=0.8)
            ax[1, j].plot(t[m], tau[m], color=C["blue"], lw=0.8, alpha=0.7, label="measured (repeats)" if n == 0 else None)
            if n == 0: ax[1, j].plot(t[m], simulate(p, np.deg2rad(wB[m]), smooth_F(S[k + "/dF"])[m]), color=C["orange"], lw=1.2, ls="--", label="stick-slip law")
        ax[0, j].set_ylabel("ω (deg/s)"); ax[1, j].set_ylabel("τ_z (N m)"); ax[1, j].set_xlabel("time from contact (s)")
        ax[0, j].set_title(f"sweep starting at {w0} deg/s, grip force setting 100", fontsize=7)
        for a in ax[:, j]: a.axhline(0, color=C["grey"], lw=0.5)
    ax[1, 0].legend(loc="lower left", fontsize=6.5)
    fig.savefig(os.path.join(FIG, "fig7_model_overlay.png"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, "fig7_model_overlay.pdf"), bbox_inches="tight")
    # validation figure (predicted vs measured end-of-hold torque)
    fig, a = plt.subplots(figsize=(3.0, 3.0))
    for blk, c, mk in [("G", C["blue"], "o"), ("G2", C["orange"], "s"), ("C", C["green"], "^"), ("S", C["grey"], "v")]:
        v = [x for x in val if x["block"] == blk]
        a.scatter([x["meas"] for x in v], [x["pred"] for x in v], s=9, color=c, marker=mk, alpha=0.7, lw=0, label=f"block {blk} (n={len(v)})")
    lim = [-2.5, 6.5]; a.plot(lim, lim, color="k", lw=0.6); a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("measured end-of-hold torque (N m)"); a.set_ylabel("predicted (N m)"); a.legend(fontsize=6.5, loc="upper left")
    fig.savefig(os.path.join(FIG, "fig8_law_validation.png"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, "fig8_law_validation.pdf"), bbox_inches="tight")

if __name__ == "__main__":
    main()
