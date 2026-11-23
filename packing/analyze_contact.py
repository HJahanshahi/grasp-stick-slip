# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""
analyze_contact.py  -  clean per-trial metrics of the finger-closure contact transient.

  python3 analyze_contact.py contact_pack.npz   ->  metrics.csv

Definitions (all forces are baseline-subtracted, baseline = 0.3 s before the close command):
  t_on      onset: chaser |dF| > 2 N (first sample after the close command)
  lagB_ms   target-arm onset minus chaser onset
  Fpk       peak chaser |dF| within 150 ms after onset;  t_pk_ms its time after onset
  rise_ms   10 %-90 % rise time to the peak
  J150      impulse |int dF dt| over 150 ms after onset (N s)
  L150      torque impulse over the same window (N m s)
  plateau   mean |dF| over the last 100 ms before release (N)
  w_B       target wrist-3 rate at onset (rad/s); v_rim = w_B * R_RIM
  finger_rate  mean finger position rate before onset (counts/s), from the gripper log
  tact_lag_ms  first tactile hit (8 sigma) minus t_on;  first_finger 1..3 (0 = none)
"""
import sys, json, csv
import numpy as np

R_RIM = 0.0737
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

def main():
    z = np.load(sys.argv[1] if len(sys.argv) > 1 else "contact_pack.npz")
    rows = []
    for k in sorted(set(x.split("/")[0] for x in z.files)):
        m = json.loads(str(z[k + "/meta"])); r = m["row"]
        ev = {}
        for n, t in m["events"]: ev.setdefault(n.split()[0], t)
        if "close_cmd" not in ev or "release" not in ev: continue
        tc, tr = ev["close_cmd"], ev["release"]
        def arm(name):
            X = z[k + "/" + name]; t0, trec = z[k + "/" + name + "_t0"]
            t = X[:, 0] - t0 + trec; F = X[:, 25:31]
            b = F[(t > tc - 0.3) & (t < tc)].mean(0)
            return t, F - b, X
        tA, dFA, XA = arm("armA"); tB, dFB, XB = arm("armB")
        fa = np.linalg.norm(dFA[:, :3], axis=1); fb = np.linalg.norm(dFB[:, :3], axis=1)
        ia = np.where((tA > tc) & (tA < tr) & (fa > 2.0))[0]
        if not len(ia):
            rows.append(dict(trial=r["trial"], split=r["split"], close_speed=int(r["close_speed"]), spin=int(r["spin"]), start=r["start"], contact=0)); continue
        on = tA[ia[0]]
        ib = np.where((tB > tc) & (tB < tr) & (fb > 2.0))[0]
        lagB = 1000 * (tB[ib[0]] - on) if len(ib) else np.nan
        w = (tA >= on) & (tA < on + 0.15)
        kp = np.argmax(fa[w]); Fpk = fa[w][kp]; t_pk = tA[w][kp] - on
        seg = fa[w][:kp + 1]
        i10 = np.argmax(seg >= 0.1 * Fpk); i90 = np.argmax(seg >= 0.9 * Fpk)
        rise = (i90 - i10) / 500.0
        J150 = np.linalg.norm(_trapz(dFA[w][:, :3], tA[w], axis=0))
        L150 = np.linalg.norm(_trapz(dFA[w][:, 3:], tA[w], axis=0))
        pl = (tA > tr - 0.1) & (tA < tr); plateau = fa[pl].mean() if pl.any() else np.nan
        kB = np.argmin(np.abs(tB - on)); wB = XB[kB, 12]   # actual_qd wrist 3 (col 7+5)
        gl = np.array(m.get("gripper_log", []), float); frate = np.nan
        if len(gl) >= 3:
            s = (gl[:, 0] > tc) & (gl[:, 0] < on)
            if s.sum() >= 2 and gl[s, 0][-1] > gl[s, 0][0]:
                frate = np.polyfit(gl[s, 0] - gl[s, 0][0], gl[s, 4:7].mean(1), 1)[0]
        T = z[k + "/tactile"]; tlag = np.nan; ff = 0
        if len(T) > 20:
            th, Xt = T[:, 0], T[:, 2:11]; rest = Xt[(th > tc - 1) & (th < tc)]
            if len(rest) >= 10:
                dev = np.abs(Xt - rest.mean(0)) / (rest.std(0) + 1)
                h = np.where((th > tc) & (th < tr + 0.2) & (dev.max(1) > 8))[0]
                if len(h): tlag = 1000 * (th[h[0]] - on); ff = int(np.argmax(dev[h[0]].reshape(3, 3).max(1))) + 1
        rows.append({kk: (float(v) if isinstance(v, (np.floating, float, np.integer)) else v) for kk, v in dict(
            trial=r["trial"], split=r["split"], close_speed=int(r["close_speed"]), spin=int(r["spin"]), start=r["start"], contact=1,
            t_on=on, onset_after_close_ms=1000 * (on - tc), lagB_ms=lagB, Fpk=Fpk, t_pk_ms=1000 * t_pk, rise_ms=1000 * rise,
            J150=J150, L150=L150, plateau=plateau, w_B=wB, v_rim=wB * R_RIM, finger_rate=frate,
            tact_lag_ms=tlag, first_finger=ff).items()})
    keys = sorted(set().union(*[x.keys() for x in rows]), key=lambda s: (s != "trial", s))
    with open("metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    json.dump(rows, open("metrics.json", "w"), indent=1)
    c = [x for x in rows if x["contact"]]
    print(f"{len(rows)} trials, {len(c)} contacts -> metrics.csv")
    for cs in sorted(set(x["close_speed"] for x in c)):
        for st in ("open", "half"):
            g = [x for x in c if x["close_speed"] == cs and x["start"] == st]
            if g: print(f"cs={cs:3d} {st:5s} n={len(g):2d}  Fpk {np.mean([x['Fpk'] for x in g]):5.1f}+-{np.std([x['Fpk'] for x in g]):4.1f} N  "
                        f"rise {np.mean([x['rise_ms'] for x in g]):4.0f} ms  J150 {np.mean([x['J150'] for x in g]):4.2f} N s  plateau {np.mean([x['plateau'] for x in g]):5.1f} N")

if __name__ == "__main__":
    main()
