# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""
pipeline.py  -  one pass over both sessions: per-trial metrics and aligned signals for the figures.

  python3 pipeline.py  ->  out/metrics.csv, out/signals.npz

Conventions (all forces are the chaser's, baseline = mean over 0.3 s before the close command):
  dF        force rise |F - F0| (N);  tau_z  torque about the chaser tool z axis (N m), baseline removed
  t_on      onset: first sample after the close command with dF > 2 N
  hold end  the spin_down event (session 2) or the release command (session 1)
  closure   Fpk / t_pk / rise10-90 / J150 over 150 ms after onset;  plateau = mean dF over the last 100 ms of the hold
  steady    mean tau_z and dF over the last 0.5 s of the hold (0.1 s when the hold is shorter), wB = target wrist-3 rate
  sweeps    (t - t_on, wB, tau_z, dF) at 500 Hz, plus loop metrics at +w, -w and the two zero crossings
  repeated  per-grasp closure metrics for ngrasp > 1
"""
import os, json, csv
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
PACKS = [("s1", os.path.join(HERE, "data", "pack.npz")),
         ("s2", os.path.join(HERE, "data2", "pack_s2.npz")),
         ("s2free", os.path.join(HERE, "data2", "pack_s2bad.npz"))]
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
R_RIM = 0.0737
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

def events(meta):
    """first occurrence of each event name, and the full ordered list"""
    ev = {}
    for n, t in meta["events"]: ev.setdefault(n.split()[0], t)
    return ev, meta["events"]

def arm(z, k, name):
    X = z[f"{k}/{name}"].astype(float); t0, trec = z[f"{k}/{name}_t0"]
    return X[:, 0] - t0 + trec, X

def load_trial(z, k):
    meta = json.loads(str(z[k + "/meta"])); row = meta["row"]
    tA, XA = arm(z, k, "armA"); tB, XB = arm(z, k, "armB")
    ev, evlist = events(meta)
    tc = ev["close_cmd"]
    F = XA[:, 25:31]; pre = (tA > tc - 0.3) & (tA < tc)
    F0 = F[pre].mean(0)
    zA = R.from_rotvec(XA[0, 16:19]).as_matrix()[:, 2]
    dF = np.linalg.norm(F[:, :3] - F0[:3], axis=1)
    tau = (F[:, 3:] - F0[3:]) @ zA
    wB = np.interp(tA, tB, XB[:, 12])                         # wrist-3 rate on the chaser clock
    prot = bool(((XA[:, 49].astype(int) & 1024) > 0).any() or ((XB[:, 49].astype(int) & 1024) > 0).any())
    return dict(meta=meta, row=row, ev=ev, evlist=evlist, tA=tA, dF=dF, tau=tau, wB=wB, F0=F0, XA=XA, tB=tB, XB=XB, prot=prot)

def onset(tA, dF, tc, te, hi=5.0, lo=2.0, sustain=0.10):
    """contact onset: the first time dF exceeds `hi` and stays above it for `sustain` s (this rejects the
    short reaction of the finger motors on the sensor when the fingers start from open), then the last
    crossing of `lo` before that time."""
    n = int(sustain * 500)
    idx = np.where((tA > tc) & (tA < te) & (dF > hi))[0]
    for i in idx:
        if i + n < len(dF) and (dF[i:i + n] > hi).all():
            j = i
            while j > 0 and dF[j] > lo and tA[j] > tc: j -= 1
            return float(tA[j])
    return None

def closure_metrics(tA, dF, t_on, t_end):
    w = (tA >= t_on) & (tA < t_on + 0.15)
    kp = int(np.argmax(dF[w])); Fpk = float(dF[w][kp]); t_pk = float(tA[w][kp] - t_on)
    seg = dF[w][:kp + 1]
    i10 = int(np.argmax(seg >= 0.1 * Fpk)); i90 = int(np.argmax(seg >= 0.9 * Fpk))
    J150 = float(_trapz(dF[w], tA[w]))
    pl = (tA > t_end - 0.1) & (tA < t_end)
    return dict(Fpk=Fpk, t_pk_ms=1000 * t_pk, rise_ms=1000 * (i90 - i10) / 500.0, J150=J150,
                plateau=float(dF[pl].mean()) if pl.any() else np.nan)

def tactile_lag(z, k, tc, t_on, t_end):
    T = z[k + "/tactile"]
    if len(T) < 20: return np.nan, 0
    th, Xt = T[:, 0], T[:, 2:11]; rest = Xt[(th > tc - 1) & (th < tc)]
    if len(rest) < 10: return np.nan, 0
    dev = np.abs(Xt - rest.mean(0)) / (rest.std(0) + 1)
    h = np.where((th > tc) & (th < t_end + 0.2) & (dev.max(1) > 8))[0]
    if not len(h): return np.nan, 0
    return 1000 * (th[h[0]] - t_on), int(np.argmax(dev[h[0]].reshape(3, 3).max(1))) + 1

def finger_rate(meta, tc, t_on):
    gl = np.array(meta.get("gripper_log", []), float)
    if len(gl) < 3: return np.nan
    s = (gl[:, 0] > tc) & (gl[:, 0] < t_on)
    if s.sum() < 2 or gl[s, 0][-1] <= gl[s, 0][0]: return np.nan
    return float(np.polyfit(gl[s, 0] - gl[s, 0][0], gl[s, 4:7].mean(1), 1)[0])

def main():
    rows, sig = [], {}
    for sess, path in PACKS:
        if not os.path.exists(path): print("missing", path); continue
        z = np.load(path)
        for k in sorted(set(x.split("/")[0] for x in z.files)):
            d = load_trial(z, k); r = d["row"]; ev = d["ev"]
            if r.get("split") == "dry": continue
            mode = r.get("mode") or "kin"
            if sess == "s2free" and mode != "free": continue        # the bad pack also holds the invalid 238-240 runs
            base = dict(session=sess, trial=f"{sess}_{r['trial']}", block=r.get("block") or "C", mode=mode,
                        close_speed=int(r["close_speed"]), spin=int(r["spin"]), start=r["start"],
                        force=int(r.get("force") or 40), hold=float(r.get("hold") or 0.25),
                        ngrasp=int(r.get("ngrasp") or 1), inertia=r.get("inertia") or "", split=r["split"],
                        prot_stop=int(d["prot"]))
            tA, dF, tau, wB = d["tA"], d["dF"], d["tau"], d["wB"]
            # grasp boundaries: (close_cmd, hold_end) per grasp
            names = [n for n, _ in d["evlist"]]; times = [t for _, t in d["evlist"]]
            closes = [t for n, t in d["evlist"] if n.startswith("close_cmd")]
            ends = []
            for tc in closes:
                after = [(n, t) for n, t in d["evlist"] if t > tc and (n.startswith("spin_down") or n.startswith("release"))]
                ends.append(after[0][1] if after else tA[-1])
            for g, (tc, te) in enumerate(zip(closes, ends)):
                t_on = onset(tA, dF, tc, te)
                if t_on is None:
                    rows.append(dict(base, grasp=g + 1, contact=0)); continue
                cm = closure_metrics(tA, dF, t_on, te)
                hold = (tA > t_on) & (tA < te); hl = te - t_on
                st = (tA > te - min(0.5, max(0.1, hl - 0.2))) & (tA < te)
                tl, ff = tactile_lag(z, k, tc, t_on, te) if g == 0 else (np.nan, 0)
                rec = dict(base, grasp=g + 1, contact=1, t_on=t_on, onset_after_close_ms=1000 * (t_on - tc),
                           hold_s=hl, wB_hold=float(np.rad2deg(wB[hold].mean())), wB_sd=float(np.rad2deg(wB[hold].std())),
                           tau_steady=float(tau[st].mean()), dF_steady=float(dF[st].mean()),
                           tau_sd=float(tau[st].std()), finger_rate=finger_rate(d["meta"], tc, t_on),
                           tact_lag_ms=tl, first_finger=ff, **cm)
                if mode in ("sweep", "sweep15"):
                    per = base["hold"] if mode == "sweep" else base["hold"] / 1.5
                    w0 = np.deg2rad(base["spin"]); rel = tA - t_on
                    seg = lambda a, b: (rel > a) & (rel < b)
                    # cosine sweep: rate = w0 cos(2 pi t / per) from the close... we align on onset (close->onset < 0.3 s)
                    s_half = seg(per / 2 - 0.15, per / 2 + 0.15); s_end = seg(per - 0.15, per - 0.02)
                    zc1 = seg(per / 4 - 0.3, per / 4 + 0.3) & (np.abs(np.rad2deg(wB)) < 2)
                    zc2 = seg(3 * per / 4 - 0.3, 3 * per / 4 + 0.3) & (np.abs(np.rad2deg(wB)) < 2)
                    rec.update(tau_start=float(tau[seg(0.3, 0.6)].mean()), tau_half=float(tau[s_half].mean()),
                               tau_end=float(tau[s_end].mean()), tau_zc1=float(tau[zc1].mean()) if zc1.any() else np.nan,
                               tau_zc2=float(tau[zc2].mean()) if zc2.any() else np.nan, dF_start=float(dF[seg(0.3, 0.6)].mean()),
                               dF_half=float(dF[s_half].mean()))
                    m = (rel > -0.3) & (rel < hl + 0.3)
                    sig[f"{base['trial']}/t"] = rel[m].astype(np.float32); sig[f"{base['trial']}/wB"] = np.rad2deg(wB[m]).astype(np.float32)
                    sig[f"{base['trial']}/tau"] = tau[m].astype(np.float32); sig[f"{base['trial']}/dF"] = dF[m].astype(np.float32)
                elif g == 0 and mode == "kin":
                    t_rel = next((t for n, t in d["evlist"] if n.startswith("release") and t > tc), te)
                    m = (tA > tc - 0.3) & (tA < t_rel + 0.8)
                    sig[f"{base['trial']}/events"] = np.array([tc - t_on, te - t_on, t_rel - t_on], np.float32)   # close cmd, hold end, release
                    sig[f"{base['trial']}/t"] = (tA[m] - t_on).astype(np.float32); sig[f"{base['trial']}/dF"] = dF[m].astype(np.float32)
                    sig[f"{base['trial']}/tau"] = tau[m].astype(np.float32); sig[f"{base['trial']}/wB"] = np.rad2deg(wB[m]).astype(np.float32)
                    sig[f"{base['trial']}/dFB"] = np.linalg.norm(d["XB"][:, 25:28] - d["XB"][(d["tB"] > tc - 0.3) & (d["tB"] < tc), 25:28].mean(0), axis=1)[np.minimum(np.searchsorted(d["tB"], tA[m]), len(d["tB"]) - 1)].astype(np.float32)
                    gl = np.array(d["meta"].get("gripper_log", []), float)
                    if len(gl): sig[f"{base['trial']}/grip"] = np.column_stack([gl[:, 0] - t_on, gl[:, 4:7]]).astype(np.float32)
                    T = z[k + "/tactile"]
                    if len(T): sig[f"{base['trial']}/tact"] = np.column_stack([T[:, 0] - t_on, T[:, 2:11]]).astype(np.float32)
                if mode == "free":
                    fl = np.array(d["meta"].get("free_log", []), float)
                    if len(fl): sig[f"{base['trial']}/free"] = np.column_stack([fl[:, 0] - t_on, fl[:, 1:4]]).astype(np.float32)
                    if g == 0:                       # 500 Hz chaser force/torque and target rate, for the force-scaled law
                        m = (tA > t_on - 0.3) & (tA < te + 0.5)
                        sig[f"{base['trial']}/t"] = (tA[m] - t_on).astype(np.float32); sig[f"{base['trial']}/dF"] = dF[m].astype(np.float32)
                        sig[f"{base['trial']}/tau"] = tau[m].astype(np.float32); sig[f"{base['trial']}/wB"] = np.rad2deg(wB[m]).astype(np.float32)
                rows.append(rec)
    keys = []
    for r_ in rows:
        for kk in r_:
            if kk not in keys: keys.append(kk)
    with open(os.path.join(OUT, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    np.savez_compressed(os.path.join(OUT, "signals.npz"), **sig)
    c = [x for x in rows if x["contact"]]
    print(f"{len(rows)} grasp records, {len(c)} contacts; sessions:", {s: sum(1 for x in rows if x['session'] == s) for s in ('s1', 's2', 's2free')})
    print("blocks:", {b: sum(1 for x in c if x['block'] == b) for b in sorted(set(x['block'] for x in c))})

if __name__ == "__main__":
    main()
