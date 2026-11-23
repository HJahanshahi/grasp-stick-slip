# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""law_free_body.py - replays the six-axis free-body captures with the grasp-torque law fitted on the one-axis sweeps.
No parameter is refitted. Each trial is simulated about the spin axis from its measured spin at first contact,
with its inertia, hand policy and hold time (same replay as one_axis/fig7_replay.py).

    python law_free_body.py [dataset] [sign]

dataset: path without extension (default data/free_body_trials). sign = +1 (default) or -1 maps the free-body
spin sign to the rim-rate sign of the law. Output: law_free_body.csv (one row per trial) and a printed summary."""
import os, sys, csv, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "free_body_trials")
SG = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
FMODE = "model"
# ---- law (one_axis/out/law_params.json) and grasp-force model (one_axis/out/force_model.json)
P = np.array([18.226839128435657, 7.576894038658541, 2.5894049765172324, 7.557897412443152e-27, 0.5641941560541658, 4.4045840701688315,
              1.8077901051997332, 1.434108483334086, 40.0])
QF = (41.820044942353064, 0.0731162526451035, 0.670205705426857, 1.253784874232271); F_MAX = 75.0; F_REF = 50.0; K_HW = 24.3; DT = 1 / 512


def rotm(rv):
    th = np.linalg.norm(rv, axis=1); k = rv / np.maximum(th, 1e-12)[:, None]
    K = np.zeros((len(rv), 3, 3)); K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 2], k[:, 1], k[:, 2], -k[:, 0], -k[:, 1], k[:, 0]
    return np.eye(3)[None] + np.sin(th)[:, None, None] * K + (1 - np.cos(th))[:, None, None] * (K @ K)


def grasp_force(t, th):
    F0, Tc, aw, as_ = QF
    return min(F0 * (1 - np.exp(-max(t, 0) / Tc)) + aw * max(-th, 0) + as_ * max(th, 0), F_MAX)


def replay(I, w_c, mode, T_hold, T_op, t_after=1.0, Fmeas=None):
    """returns columns: time from contact, spin (deg/s), hand angle (deg), torque. Law frame."""
    k, cp, cn = P[:3], P[3:6], P[6:9]; D = np.sqrt(K_HW * I)
    w = np.deg2rad(w_c); e = np.zeros(3); th = 0.0; ah = 0.0; tau = 0.0; F_rel = 0.0; out = []
    for i in range(int((T_hold + T_op + t_after) / DT)):
        t = i * DT
        if t < T_hold: F = grasp_force(t, th) if Fmeas is None else float(np.interp(t, Fmeas[0], Fmeas[1])); F_rel = F
        elif t < T_hold + T_op: F = F_rel * (1 - (t - T_hold) / T_op)
        else: F = 0.0
        if F > 0:
            wh = 0.0
            if mode == "damp" and t < T_hold:
                wh = float(np.clip(-tau / D, -np.deg2rad(40), np.deg2rad(40)))
                if abs(ah + np.rad2deg(wh) * DT) > 45: wh = 0.0
            wr = w - wh; th += np.rad2deg(wr) * DT; ah += np.rad2deg(wh) * DT
            s_ = max(F, 1e-3) / F_REF; e = np.clip(e - s_ * k * wr * DT, -s_ * cp, s_ * cn); tau = e.sum()
        else: e[:] = 0.0; tau = 0.0
        w += tau / I * DT; out.append((t, np.rad2deg(w), ah, tau))
    return np.array(out)


def metrics(t, w, tau, T_hold, s0):
    """same numbers from a measured or a simulated spin trace (t from contact, w in deg/s)"""
    h = t < T_hold; sg = np.sign(s0); dt = np.gradient(t)
    k = np.flatnonzero(h & (w * sg <= 0)); ks = k[0] if len(k) else None
    o = dict(t_stop=float(t[ks]) if ks is not None else np.nan, slip_angle=float(np.sum((np.abs(w) * dt)[:ks if ks is not None else h.sum()])),
             rebound=float((-sg * w[ks:h.sum()]).max()) if ks is not None else np.nan, w_rel=float(w[h][-1]),
             w_end=float(w[(t > T_hold + 0.6) & (t < T_hold + 1.0)].mean()) if (t > T_hold + 0.6).any() else np.nan,
             tau_slip=float(np.abs(tau[:ks if ks is not None else h.sum()]).mean()) if tau is not None else np.nan)
    # ring-down: zero crossings and successive swing peaks while held
    zc = np.flatnonzero(h[1:] & (np.sign(w[1:]) != np.sign(w[:-1])) & (np.abs(w[1:] - w[:-1]) > 0))
    zc = [z for j, z in enumerate(zc) if j == 0 or t[z] - t[zc[j - 1]] > 0.15]
    pk = [float(np.abs(w[a:b]).max()) for a, b in zip(zc[:-1], zc[1:])]
    o["half_period"] = float(np.mean(np.diff(t[zc]))) if len(zc) > 1 else np.nan
    o["swing1"] = pk[0] if pk else np.nan
    o["decay"] = float(np.mean([b / a for a, b in zip(pk[:-1], pk[1:]) if a > 1.0])) if len(pk) > 1 and pk[0] > 1.0 else np.nan
    return o


def main():
    Z = np.load(DS + ".npz", allow_pickle=True); cols = list(Z["em_cols"]); ic = {c: j for j, c in enumerate(cols)}
    rows = [r for r in csv.DictReader(open(DS + "_index.csv")) if r["use"] == "1" and r["mode"] == "grasp"]
    out = []
    for r in rows:
        E = Z[r["trial"] + "/em"]; ev = {}
        for n, tt in json.loads(str(Z[r["trial"] + "/events"])): ev.setdefault(n.split()[0], tt)
        t = E[:, 0]; g = E[:, ic["gate"]] > 0; k0 = int(np.argmax(g)); t_c = t[k0]; t_rel = ev["release"]
        w = np.rad2deg(E[:, ic["wz"]]); post = (t >= t_rel) & g
        T_hold = t_rel - t_c; T_op = float(t[post].max() - t_rel) if post.any() else 0.1
        I = float(r["I_zz"]); s0 = float(r["spin"]); w_c = float(w[k0])
        tau = np.einsum("nji,nj->ni", rotm(E[:, ic["rx"]:ic["rz"] + 1]), E[:, ic["Tx"]:ic["Tz"] + 1])[:, 2]      # contact torque about the body spin axis
        m = t >= t_c; tm = t[m] - t_c
        Fm = None
        if FMODE == "meas":
            Fn = np.linalg.norm(E[m, ic["Fx"]:ic["Fz"] + 1], axis=1); Fm = (tm, np.convolve(Fn, np.ones(7) / 7, "same"))
        M = metrics(tm, w[m], tau[m], T_hold, s0)
        S = replay(I, SG * w_c, r["hand"], T_hold, max(T_op, DT), Fmeas=Fm); Sm = metrics(S[:, 0], SG * S[:, 1], S[:, 3], T_hold, s0)
        rec = dict(trial=r["trial"], label=r["label"], hand=r["hand"], spin=s0, I=I, nut=r["nut"], hold=round(T_hold, 2), w_c=round(w_c, 2))
        for k_ in M: rec["m_" + k_] = round(M[k_], 3); rec["p_" + k_] = round(Sm[k_], 3)
        out.append(rec)
    with open(os.path.join(HERE, "law_free_body.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys())); wr.writeheader(); wr.writerows(out)
    A = lambda g, k_: np.array([x[k_] for x in g], float)
    for lab in ("grasp_centred", "grasp_ringdown", "grasp_wobble_tracked", "grasp_offcentre"):
        print(f"\n== {lab}   (measured / law replay, means)")
        print("  hand   I     spin   n   stop time (s)     slip angle (deg)   rebound (deg/s)   spin after release (deg/s)   slip torque (N m)")
        for hand in ("rigid", "damp"):
            for I in sorted({x["I"] for x in out if x["label"] == lab}):
                for sp in sorted({x["spin"] for x in out if x["label"] == lab}):
                    g = [x for x in out if x["label"] == lab and x["hand"] == hand and x["I"] == I and x["spin"] == sp]
                    if not g: continue
                    f = lambda k_: f"{np.nanmean(A(g, 'm_' + k_)):6.2f} / {np.nanmean(A(g, 'p_' + k_)):6.2f}"
                    print(f"  {hand:5s} {I:5.2f} {sp:+5.0f} {len(g):3d}   {f('t_stop')}   {f('slip_angle')}    {f('rebound')}    {f('w_end')}            {f('tau_slip')}")
        g = [x for x in out if x["label"] == lab]
        if not g: continue
        for k_, u in (("t_stop", "s"), ("slip_angle", "deg"), ("rebound", "deg/s"), ("w_end", "deg/s"), ("tau_slip", "N m")):
            a, b = A(g, "m_" + k_), A(g, "p_" + k_); ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 3: continue
            e = b[ok] - a[ok]
            print(f"  {k_:11s} n={ok.sum():3d}  RMS error {np.sqrt(np.mean(e**2)):.2f} {u}, bias {e.mean():+.2f}, corr {np.corrcoef(a[ok], b[ok])[0,1]:.2f}, measured mean {np.mean(np.abs(a[ok])):.2f}")
    g = [x for x in out if x["label"] == "grasp_ringdown"]
    print("\n== ring-down (8 s rigid hold): measured / law")
    for x in g: print(f"  {x['trial']} I={x['I']} spin={x['spin']:+.0f}  half period {x['m_half_period']:.2f} / {x['p_half_period']:.2f} s   first swing {x['m_swing1']:.1f} / {x['p_swing1']:.1f} deg/s   swing ratio {x['m_decay']:.2f} / {x['p_decay']:.2f}")


if __name__ == "__main__":
    main()
