# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""force_model.py - grasp force for simulations, where no force is measured.
F(t, theta) = F0 (1 - exp(-t / T_c)) + a_w max(-theta, 0) + a_s max(theta, 0)
t: time since contact (s); theta: rim angle turned since contact (deg), negative = wedging direction.
Fitted on the 54 constant-rate holds at closing speed 255 (Fig 4). Output: out/force_model.json"""
import os, csv, json
import numpy as np
from scipy.optimize import least_squares
F_MAX = 75.0      # N, the largest grasp force seen in the holds; the model is not extrapolated beyond it
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")

def grasp_force(t, theta_deg, q):
    F0, Tc, aw, as_ = q
    F = F0 * (1 - np.exp(-np.maximum(t, 0) / Tc)) + aw * np.maximum(-theta_deg, 0) + as_ * np.maximum(theta_deg, 0)
    return np.minimum(F, F_MAX)

def load():
    return json.load(open(os.path.join(OUT, "force_model.json")))["q"]

if __name__ == "__main__":
    S = np.load(os.path.join(OUT, "signals.npz"))
    rows = [r for r in csv.DictReader(open(os.path.join(OUT, "metrics.csv"))) if r["contact"] == "1" and r["mode"] == "kin" and r["grasp"] == "1"
            and r["block"] in ("G", "G2") and r["close_speed"] == "255"]
    T, TH, F, ID = [], [], [], []
    for j, r in enumerate(rows):
        k = r["trial"]; t, w, dF = S[k + "/t"], S[k + "/wB"], S[k + "/dF"]; m = (t > 0) & (t < float(r["hold_s"]))
        th = np.cumsum(w[m] * np.r_[0, np.diff(t[m])]); sm = np.convolve(dF, np.ones(25) / 25, "same")[m]
        T.append(t[m][::5]); TH.append(th[::5]); F.append(sm[::5]); ID.append(np.full(len(th[::5]), j))
    T, TH, F, ID = map(np.concatenate, (T, TH, F, ID))
    sol = least_squares(lambda q: grasp_force(T, TH, q) - F, [45, 0.08, 0.5, 0.8], bounds=([0, 0.005, 0, 0], [150, 1, 5, 5]), loss="soft_l1", f_scale=3)
    q = sol.x; e = grasp_force(T, TH, q) - F
    per = [np.sqrt(np.mean(e[ID == j] ** 2)) for j in range(len(rows))]
    print(f"F0 {q[0]:.1f} N, T_c {1000*q[1]:.0f} ms, a_wedging {q[2]:.2f} N/deg, a_slip {q[3]:.2f} N/deg; RMSE {np.sqrt(np.mean(e**2)):.1f} N (median per trial {np.median(per):.1f} N)")
    json.dump(dict(q=q.tolist(), form="F0 (1 - exp(-t/Tc)) + a_w max(-theta,0) + a_s max(theta,0), theta in deg", rmse=float(np.sqrt(np.mean(e**2))), n_trials=len(rows)),
              open(os.path.join(OUT, "force_model.json"), "w"), indent=1)
