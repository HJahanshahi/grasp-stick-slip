# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""fig7_replay.py - the force-scaled law replays the closed-loop hardware captures (rigid hand vs damping hand).
Each hardware trial is simulated from its spin at first contact, with its inertia, policy and release time.
Grasp force from force_model (no force is logged in the emulator); at release the force ramps to zero over the
measured opening time (release command to last contact)."""
import os, json, glob, re
import numpy as np
from fit_law import N_EL, DT, F_REF
from force_model import grasp_force, load as load_force
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
P = np.array(json.load(open(os.path.join(OUT, "law_params.json")))["params"]); QF = load_force()
B = np.load(os.path.join(OUT, "law_bootstrap.npy"))
K_HW = 24.3                      # stiffness used by the damping hand on hardware (old law)

def trials():
    out = []
    for d in sorted(glob.glob(os.path.join(HERE, "data3", "raw", "contact_data", "trial_*"))):
        mp = os.path.join(d, "meta.json")
        if not os.path.exists(mp): continue
        m = json.load(open(mp)); r = m["row"]
        if r.get("mode") not in ("rigid", "damp") or "free_log" not in m or r["split"] == "dry": continue
        L = np.array(m["free_log"], float); ev = {n.split()[0]: t for n, t in m["events"]}
        t_on = next(t for n, t in m["events"] if n.startswith("free_spin_on")); t_rel = ev["release"] - t_on
        t = L[:, 0] - t_on; c = L[:, 5] > 0
        w_end = float(re.search(r"w_end=(-?[\d.]+)", " ".join(n for n, _ in m["events"])).group(1))
        i0 = int(np.argmax(c)); post = (t >= t_rel) & c
        out.append(dict(trial=r["trial"], mode=r["mode"], I=float(r["inertia"]), w0=float(r["spin"]), t=t, w=np.rad2deg(L[:, 3]), t_c=t[i0], w_c=float(np.rad2deg(L[i0, 3])),
                        t_rel=t_rel, t_open=float(t[post].max() - t_rel) if post.any() else 0.1, w_end=w_end))
    return out

def replay(x, p=P, t_after=1.0, delay=0.0):
    """simulate one capture; returns (time from contact, target spin in deg/s, hand angle in deg)"""
    I = x["I"]; k = p[:N_EL]; cp = p[N_EL:2 * N_EL]; cn = p[2 * N_EL:3 * N_EL]; D = np.sqrt(K_HW * I)
    w = np.deg2rad(x["w_c"]); wh = 0.0; e = np.zeros(N_EL); th = 0.0; ah = 0.0; tau = 0.0
    T_hold = x["t_rel"] - x["t_c"]; T_op = max(x["t_open"], DT); out = []; nd = int(round(delay / DT)); buf = [0.0] * (nd + 1)
    n = int((T_hold + T_op + t_after) / DT)
    for i in range(n):
        t = i * DT
        if t < T_hold: F = float(grasp_force(t, th, QF))
        elif t < T_hold + T_op: F = F_rel * (1 - (t - T_hold) / T_op)
        else: F = 0.0
        if i and t >= T_hold and t - DT < T_hold: pass
        if t < T_hold: F_rel = F
        if F > 0:
            if x["mode"] == "damp" and t < T_hold:
                wh = float(np.clip(-tau / D, -np.deg2rad(40), np.deg2rad(40)))
                if abs(ah + np.rad2deg(wh) * DT) > 45: wh = 0.0
            else: wh = 0.0
            wr = w - wh; th += np.rad2deg(wr) * DT; ah += np.rad2deg(wh) * DT
            s_ = max(F, 1e-3) / F_REF; e = np.clip(e - s_ * k * wr * DT, -s_ * cp, s_ * cn); tau = e.sum()
        else: e[:] = 0.0; tau = 0.0
        buf.append(tau); w += buf.pop(0) / I * DT
        out.append((t, np.rad2deg(w), ah, tau if F > 0 else np.nan))
    return np.array(out)

if __name__ == "__main__":
    T = trials()
    print(len(T), "trials")
    for x in T:
        s = replay(x); x["w_sim"] = s[-1, 1]
        print(f"{x['mode']:5s} I={x['I']} w0={x['w0']:+.0f} w_c={x['w_c']:+6.1f} hold={x['t_rel']-x['t_c']:.2f} open={x['t_open']:.2f}  w_end hw {x['w_end']:+6.1f}  sim {x['w_sim']:+6.1f}")
    for p in ("rigid", "damp"):
        for I in (0.5, 1.0, 2.0):
            g = [x for x in T if x["mode"] == p and x["I"] == I]
            print(p, I, "hw %.1f sim %.1f" % (np.mean([abs(x["w_end"]) for x in g]), np.mean([abs(x["w_sim"]) for x in g])))

def figure():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import figs                                                     # shared style
    PC = {"rigid": "#D55E00", "damp": "#0072B2"}; PN = {"rigid": "rigid hand", "damp": "damping hand"}
    T = trials()
    for x in T: x["sim"] = replay(x); x["w_sim"] = x["sim"][-1, 1]
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.55, wspace=0.32)
    ax = [fig.add_subplot(gs[0, j]) for j in range(3)]
    for j, I in enumerate((0.5, 1.0, 2.0)):
        a = ax[j]
        for p in ("rigid", "damp"):
            x = next(y for y in T if y["mode"] == p and y["I"] == I and y["w0"] == 15.0)
            m = (x["t"] >= x["t_c"]) & (x["t"] <= x["t_rel"] + 1.0)
            a.plot(x["t"][m] - x["t_c"], x["w"][m], color=PC[p], lw=0.9)
            a.plot(x["sim"][:, 0], x["sim"][:, 1], color=PC[p], lw=0.9, ls=(0, (3, 1.5)), alpha=0.9)
            a.axvline(x["t_rel"] - x["t_c"], color=PC[p], lw=0.5, ls=":")
        a.axhline(0, color="#6b6b6b", lw=0.5); a.set_title(f"I = {I} kg m²", fontsize=7.5); a.set_xlabel("time from contact (s)"); a.set_xlim(0, 3.3)
        a.text(-0.16 if j == 0 else -0.1, 1.08, "(" + "abc"[j] + ")", transform=a.transAxes, fontsize=8, fontweight="bold")
    lo = min(a.get_ylim()[0] for a in ax); hi = max(a.get_ylim()[1] for a in ax)
    for a in ax: a.set_ylim(lo, hi)
    ax[0].set_ylabel("target spin (deg/s)")
    # (d) spin after release, every trial
    a = fig.add_subplot(gs[1, 0:2])
    for p in ("rigid", "damp"):
        for I, mk in ((0.5, "o"), (1.0, "s"), (2.0, "^")):
            g = [x for x in T if x["mode"] == p and x["I"] == I]
            a.scatter([x["w_end"] for x in g], [x["w_sim"] for x in g], s=16, marker=mk, color=PC[p] if I > 0.5 else "none", edgecolor=PC[p], lw=0.8, zorder=3)
    lim = [-12, 18]; a.plot(lim, lim, color="k", lw=0.6); a.set_xlim(lim); a.set_ylim(lim); a.set_aspect("equal")
    a.axhline(0, color="#6b6b6b", lw=0.5); a.axvline(0, color="#6b6b6b", lw=0.5)
    a.set_xlabel("spin after release, hardware (deg/s)"); a.set_ylabel("spin after release,\nlaw replay (deg/s)")
    a.text(-0.16, 1.04, "(d)", transform=a.transAxes, fontsize=8, fontweight="bold")
    # summary text: I >= 1
    def mean_abs(p, key): return np.mean([abs(x[key]) for x in T if x["mode"] == p and x["I"] >= 1])
    rb = []
    for b in B:
        r = [abs(replay(x, p=b)[-1, 1]) for x in T if x["I"] >= 1]
        md = [x["mode"] for x in T if x["I"] >= 1]
        rb.append((np.mean([v for v, m_ in zip(r, md) if m_ == "rigid"]), np.mean([v for v, m_ in zip(r, md) if m_ == "damp"])))
    rb = np.array(rb); q = np.percentile(rb, [5, 95], axis=0)
    txt = (f"I ≥ 1 kg m², mean |spin after release|\n"
           f"rigid: hardware {mean_abs('rigid','w_end'):.1f}, replay {mean_abs('rigid','w_sim'):.1f} ({q[0,0]:.1f} to {q[1,0]:.1f}) deg/s\n"
           f"damping: hardware {mean_abs('damp','w_end'):.1f}, replay {mean_abs('damp','w_sim'):.1f} ({q[0,1]:.1f} to {q[1,1]:.1f}) deg/s")
    ta = fig.add_subplot(gs[1, 2]); ta.axis("off")
    H = [matplotlib.lines.Line2D([], [], color=PC["rigid"], lw=1.2), matplotlib.lines.Line2D([], [], color=PC["damp"], lw=1.2),
         matplotlib.lines.Line2D([], [], color="k", lw=0.9), matplotlib.lines.Line2D([], [], color="k", lw=0.9, ls=(0, (3, 1.5))),
         matplotlib.lines.Line2D([], [], color="k", lw=0.6, ls=":"),
         matplotlib.lines.Line2D([], [], ls="", marker="o", mfc="none", mec="k", ms=4), matplotlib.lines.Line2D([], [], ls="", marker="s", mfc="k", mec="k", ms=4),
         matplotlib.lines.Line2D([], [], ls="", marker="^", mfc="k", mec="k", ms=4)]
    ta.legend(H, ["rigid hand", "damping hand", "hardware", "law replay", "release command", "I = 0.5 kg m² (emulator limit)", "I = 1 kg m²", "I = 2 kg m²"],
              loc="center left", fontsize=6.5, handlelength=1.8, borderaxespad=0, labelspacing=0.9)
    figs.save(fig, "fig7_law_replay")
    e = np.array([x["w_sim"] - x["w_end"] for x in T if x["I"] >= 1])
    print(txt); print(f"I>=1 signed error RMS {np.sqrt(np.mean(e**2)):.1f} deg/s; sign agreement {np.mean([np.sign(x['w_sim']) == np.sign(x['w_end']) for x in T if x['I'] >= 1 and abs(x['w_end']) > 1]):.2f}")
