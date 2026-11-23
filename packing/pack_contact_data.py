# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""
pack_contact_data.py  -  compress the contact trials into one file for analysis.

  python3 pack_contact_data.py contact_data  ->  contact_pack.npz  (plus impacts.csv from extract_impacts.py)

Per trial it keeps: meta.json, the tactile stream, the gripper log, and both arms' 500 Hz
records from 1.0 s before the close command to 1.0 s after the release.
"""
import os, sys, json, glob, csv
import numpy as np

def read_csv(path):
    with open(path) as f:
        rd = csv.reader(f); hdr = next(rd); rows = [r for r in rd if r]
    return hdr, np.array(rows, float)

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "contact_data"
    pack = {}; n = 0
    for d in sorted(glob.glob(os.path.join(root, "trial_*"))):
        try:
            meta = json.load(open(os.path.join(d, "meta.json")))
            ev = {}
            for name, t in meta["events"]: ev.setdefault(name.split()[0], t)
            t_rec = meta["events"][0][1]
            lo = ev.get("close_cmd", t_rec) - 1.0; hi = ev.get("release", ev.get("record_stop", t_rec)) + 1.0
            key = os.path.basename(d)
            for arm in ("armA", "armB"):
                hdr, X = read_csv(os.path.join(d, f"{arm}_500Hz.csv"))
                t = X[:, 0] - X[0, 0] + t_rec
                m = (t >= lo) & (t <= hi)
                pack[f"{key}/{arm}"] = X[m].astype(np.float32)
                pack[f"{key}/{arm}_t0"] = np.array([X[0, 0], t_rec])
                pack[f"{key}/{arm}_hdr"] = np.array(hdr)
            tj = os.path.join(d, "tactile.json")
            pack[f"{key}/tactile"] = np.array(json.load(open(tj)), float) if os.path.exists(tj) else np.zeros((0, 11))
            pack[f"{key}/meta"] = np.array(json.dumps(meta))
            n += 1
        except Exception as e:
            print(f"{os.path.basename(d)}: skipped ({e})")
    np.savez_compressed("contact_pack.npz", **pack)
    print(f"packed {n} trials -> contact_pack.npz  ({os.path.getsize('contact_pack.npz')/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
