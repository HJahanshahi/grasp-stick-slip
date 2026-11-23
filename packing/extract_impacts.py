# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""
extract_impacts.py  -  one row per trial from the capture_contact.py recordings (finger-closure contact).

  python extract_impacts.py contact_data   -> impacts.csv

Per trial:
  t_on       contact onset: tactile deviation > TACT_SIGMA (primary), else fast force rise on the chaser
  t_end      release command
  J_A, J_B   impulse on each arm (N s) over [t_on, t_end], force minus the pre-close baseline
  L_A, L_B   torque impulse (N m s)
  Fpk_A/B    peak force rise
  w_B        target spin rate at onset (rad/s, wrist 3), v_rim = w_B * R_RIM
  finger_rate  finger position rate before contact (counts/s) from the gripper log
  first_finger tactile finger that moved first
"""
import os, sys, json, glob, csv
import numpy as np
from scipy.spatial.transform import Rotation

F_RISE, F_RISE_WIN = 5.0, 0.040   # N, s  (fast rise on the chaser)
R_RIM = 0.0737                  # m, cone rim radius (5.8 in diameter)
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
PRE_WIN = 0.050               # s
TACT_SIGMA = 8.0

def read_rtde_csv(path):
    with open(path) as f:
        rd = csv.reader(f); hdr = next(rd); rows = [r for r in rd if r]
    X = np.array(rows, float)
    col = {}
    for i, h in enumerate(hdr):
        base = h[:-2] if h[-2:-1] == "_" and h[-1].isdigit() else h
        col.setdefault(base, []).append(i)
    get = lambda k: X[:, col[k]]
    return dict(t=get("timestamp")[:, 0], q=get("actual_q"), qd=get("actual_qd"),
                pose=get("actual_TCP_pose"), v=get("actual_TCP_speed"), F=get("actual_TCP_force"))

def trial_row(dirpath):
    meta = json.load(open(os.path.join(dirpath, "meta.json")))
    A = read_rtde_csv(os.path.join(dirpath, "armA_500Hz.csv"))
    B = read_rtde_csv(os.path.join(dirpath, "armB_500Hz.csv"))
    ev = {}
    for n, t in meta["events"]: ev.setdefault(n.split()[0], t)
    t_rec = meta["events"][0][1]
    tA = A["t"] - A["t"][0] + t_rec; tB = B["t"] - B["t"][0] + t_rec
    t_close, t_rel = ev.get("close_cmd"), ev.get("release")
    r = meta["row"]
    base = dict(trial=r["trial"], split=r["split"], close_speed=r["close_speed"], spin=r["spin"], start=r["start"])
    if t_close is None or t_rel is None: return dict(base, contact=0)
    # baselines just before the close command
    preA = (tA > t_close - 0.3) & (tA < t_close); preB = (tB > t_close - 0.3) & (tB < t_close)
    bA = A["F"][preA].mean(0); bB = B["F"][preB].mean(0)
    # tactile onset
    t_on = None; first_finger = None
    try:
        T = np.array(json.load(open(os.path.join(dirpath, "tactile.json"))), float)
        th, X = T[:, 0], T[:, 2:11]
        rest = X[(th > t_close - 1.0) & (th < t_close)]
        if len(rest) >= 10:
            mu, sd = rest.mean(0), rest.std(0) + 1.0
            dev = np.abs(X - mu) / sd
            hits = np.where((th >= t_close) & (th <= t_rel) & (dev.max(1) > TACT_SIGMA))[0]
            if len(hits):
                k = hits[0]; t_on = float(th[k]); first_finger = int(np.argmax(dev[k].reshape(3, 3).max(1))) + 1
    except Exception:
        pass
    # force-rise onset on the chaser (fallback, and reported for comparison)
    fA = np.linalg.norm(A["F"][:, :3] - bA[:3], axis=1)
    win = int(F_RISE_WIN * 500); t_on_F = None
    idx = np.where((tA >= t_close) & (tA <= t_rel))[0]
    for k in idx[win:]:
        if np.linalg.norm(A["F"][k, :3] - A["F"][k - win, :3]) > F_RISE:
            t_on_F = float(tA[k - win]); break
    if t_on is None: t_on = t_on_F
    if t_on is None: return dict(base, contact=0)
    def impulse(t, F, b, lo, hi, cols):
        m = (t >= lo) & (t <= hi)
        return _trapz(F[m][:, cols] - b[cols], t[m], axis=0) if m.sum() > 1 else np.zeros(len(cols))
    J_A = impulse(tA, A["F"], bA, t_on, t_rel, [0, 1, 2]); J_B = impulse(tB, B["F"], bB, t_on, t_rel, [0, 1, 2])
    L_A = impulse(tA, A["F"], bA, t_on, t_rel, [3, 4, 5]); L_B = impulse(tB, B["F"], bB, t_on, t_rel, [3, 4, 5])
    fB = np.linalg.norm(B["F"][:, :3] - bB[:3], axis=1)
    kB0 = np.argmin(np.abs(tB - t_on)); w_B = float(B["qd"][kB0, 5])
    # finger rate before contact from the gripper log
    finger_rate = None
    gl = np.array(meta.get("gripper_log", []), float)
    if len(gl) >= 3:
        m = (gl[:, 0] < t_on) & (gl[:, 0] > t_close)
        if m.sum() >= 2:
            tt, p = gl[m, 0], gl[m, 4:7].mean(1)
            finger_rate = float(np.polyfit(tt - tt[0], p, 1)[0]) if tt[-1] > tt[0] else None
    mA = (tA >= t_on) & (tA <= t_rel); mB = (tB >= t_on) & (tB <= t_rel)
    return dict(base, contact=1, t_on=t_on, t_on_force=t_on_F, tact_minus_force_ms=None if t_on_F is None else 1000 * (t_on - t_on_F),
                dur_ms=1000 * (t_rel - t_on), w_B=w_B, v_rim=w_B * R_RIM, finger_rate=finger_rate,
                J_A=float(np.linalg.norm(J_A)), J_B=float(np.linalg.norm(J_B)),
                L_A=float(np.linalg.norm(L_A)), L_B=float(np.linalg.norm(L_B)),
                Fpk_A=float(fA[mA].max()) if mA.any() else None, Fpk_B=float(fB[mB].max()) if mB.any() else None,
                first_finger=first_finger, online=next((n for n, t in meta["events"] if n.startswith("contact")), None))

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "contact_data"
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "trial_*"))):
        try:
            r = trial_row(d)
            if r: rows.append(r); print(f"{os.path.basename(d)}: contact={r['contact']}"
                                        + (f"  J_A={r['J_A']:.3f} N s  J_B={r['J_B']:.3f}  dur={r['dur_ms']:.0f} ms  Fpk_A={r['Fpk_A']:.1f} N  finger={r['first_finger']}" if r['contact'] else ""))
        except Exception as e:
            print(f"{os.path.basename(d)}: failed ({e})")
    if not rows: print("no trials"); return
    keys = sorted(set().union(*[r.keys() for r in rows]), key=lambda k: (k != "trial", k))
    with open("impacts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nwrote impacts.csv with {len(rows)} rows ({sum(r['contact'] for r in rows)} contacts)")

if __name__ == "__main__":
    main()
