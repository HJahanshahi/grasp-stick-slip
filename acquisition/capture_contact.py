# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""
capture_contact.py  -  synchronized capture with a measured contact (touch and release).

Builds on cubesat_pull_demo.py: same IPs, same taught poses (HOME/NEAR/APPROACH/GRASP),
same gripper class. Put this file in the same folder as cubesat_pull_demo.py,
tactile_sensor.py and config.py.

One trial (the contact is the fingers closing on the rim of the spinning cone):
  1. target spins at SPIN deg/s (wrist 3); the chaser holds its roll
  2. chaser moves NEAR -> straddle along the cone axis (slow), fingers at START (open / half)
  3. fingers close at CLOSE_SPEED; contact = tactile deviation, or fast force rise on the chaser,
     or a finger contact code; hold T_GRIP_HOLD after contact, then open
  4. retract D_RETRACT along the cone axis, spin down, unwind, back to NEAR
Both arms are recorded at 500 Hz (startFileRecording) and the tactile stream at its own rate,
for the whole trial. One folder per trial.

Usage:
  python capture_contact.py --make trials.csv          # write a randomized trial list
  python capture_contact.py trials.csv                 # run it (resumes: skips rows marked done)
  python capture_contact.py trials.csv --find-preset   # find the half-closed finger value first
"""
import os, sys, csv, json, time, random, argparse, threading
import numpy as np
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

import cubesat_pull_demo as D          # taught poses, IPs, Gripper3F, servo params
from tactile_sensor import TactileSensor

# ---------------------------------------------------------------- settings
OUT_ROOT     = "contact_data"
RTDE_HZ      = 500.0
DT           = D.DT                    # 1/125 streaming rate
SPIN_RAMP    = 1.0                     # s
T_APPROACH   = 3.0                     # s, NEAR -> straddle
AXIS_SIGN    = +1                      # set to -1 if "move to NEAR" moves the chaser TOWARDS the cone
D_NEAR       = 0.060                   # m, start of each trial, back along the cone axis from the straddle
D_ACTIVATE   = 0.100                   # m, back-off for the gripper activation sweep
D_RETRACT    = 0.060                   # m, back to NEAR after each trial
V_RETRACT    = 0.050                   # m/s
F_RISE       = 5.0                     # N, rise of the chaser force within F_RISE_WIN counts as contact
F_RISE_WIN   = 0.040                   # s
TACT_SIGMA   = 8.0                     # tactile: |x - rest| > TACT_SIGMA * std(rest) counts as contact
T_SETTLE     = 0.4                     # s, at the straddle before closing
T_CLOSE_MAX  = 3.0                     # s, max wait for contact after the close command
T_GRIP_HOLD  = 0.25                    # s, fingers stay on the rim after contact (short: the rim is turning)
# --- session 2: free-spin emulation on the target wrist (1-DOF) after contact
TORQUE_SIGN  = +1                      # chaser torque about its tool z, session-1 data: negative at +spin -> decelerates with +1
TAU_DB       = 0.03                    # N m (chaser torque noise sd is 0.005 N m; finger-offset model error ~0.05)
FINGER_FILE  = "finger_offset_A.json"  # chaser F/T torque about tool z vs finger position (finger mass), no contact
TAU_DB_B     = 0.15                    # N m, torque deadband (sensor noise)
TAU_ALPHA    = 0.70                    # EMA on the torque (~3 ms lag; torque noise is only 0.005 N m sd)
SERVO_LA_FREE  = 0.04                  # s, target servoJ lookahead in free-spin modes (0.1 s adds ~0.1 s loop delay,
SERVO_GAIN_FREE = 600                  #    i.e. negative damping of the emulated stick oscillation)
W_MAX        = np.deg2rad(50.0)        # rad/s, emulated spin cap
ACC_MAX      = 5.0                     # rad/s^2
ANG_MAX      = np.deg2rad(330.0)       # abort if the target wrist winds past this
W_UNJAM      = np.deg2rad(10.0)        # rad/s, brief positive rim motion before the release after a negative spin
T_REGRASP    = 1.2                     # s, fingers open between repeated grasps
F_CONTACT    = 3.0                     # N, chaser force rise that gates the torque integration (fingers on the rim)
BIAS_FILE    = "ft_bias_B.json"        # target F/T bias vs wrist-3 angle (gravity compensation residual)
K_GRIP       = 24.3                    # N m/rad, elastic stiffness of the grasp (fitted on the sweeps)
HAND_RATE_MAX = np.deg2rad(40.0)       # rad/s, damper mode: chaser wrist-3 rate cap
HAND_ANG_MAX = np.deg2rad(45.0)        # rad, damper mode: chaser wrist-3 excursion cap
HAND_OFFSET_FILE = "hand_offset_A.json"  # chaser F/T reading vs chaser wrist-3 angle (gripper weight), no contact
T_GATE_OPEN  = 0.25                    # s, contact torque is integrated at most this long after the open command
HAND_ACC_MAX = 4.0                     # rad/s^2, damper mode: chaser wrist-3 acceleration cap
HAND_SIGN_FILE = "hand_sign.json"      # sign linking chaser wrist-3 rotation to the measured torque
W_STOP       = np.deg2rad(2.0)         # rad/s, closed loop: stop regrasping below this emulated rate
T_LOOP_MAX   = 12.0                    # s, closed loop: give up after this time since the first contact
T_POST_FREE  = 1.0                     # s, free spin recorded after the last release
HALF_PRESET_DEFAULT = 90               # finger value for "half" if not found; --find-preset sets it
PRESET_FILE  = "finger_preset.json"
STRADDLE_FILE = "straddle.json"        # taught by --teach-straddle: joints of both arms at the straddle
RECORD_VARS  = ["timestamp", "actual_q", "actual_qd", "actual_TCP_pose", "actual_TCP_speed",
                "actual_TCP_force", "target_q", "target_qd", "actual_current",
                "safety_status_bits", "robot_status_bits"]

# ---------------------------------------------------------------- trial list
def make_trials(path, reps=3, seed=1):
    close_speed = [40, 100, 170, 255]          # Robotiq speed setting (tips ~20 .. 110 mm/s)
    spin        = [-12, -6, 0, 6, 12]          # deg/s, target spin; chaser roll fixed
    start       = ["open", "half"]             # finger position before closing
    rows = []
    for cs in close_speed:
        for sp in spin:
            for st in start:
                for r in range(reps):
                    rows.append(dict(close_speed=cs, spin=sp, start=st, rep=r))
    random.Random(seed).shuffle(rows)
    n = len(rows)
    for i, row in enumerate(rows):
        row["trial"] = i + 1
        row["split"] = "test" if i >= int(0.8 * n) else "calib"
        row["status"] = ""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "close_speed", "spin", "start", "rep", "split", "status"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {n} trials to {path}  (~{n*0.4:.0f} min at 25 s each)")

TRIAL_COLS = ["trial", "block", "mode", "close_speed", "spin", "start", "force", "hold", "inertia", "ngrasp", "rep", "split", "status"]
def make_trials2(path, seed=2):
    """Session 2. Blocks in order of value; each block is shuffled and split 80/20 calib/test.
       F  free-spin emulation, single grasp (detumble by one grasp): inertia x spin x speed
       R  repeated grasps on the emulated free spin (detumble by repeated contact)
       G  grip-force sweep, kinematic spin, 1 s hold (steady slip torque)
       S  wider spin range for the closure map (kinematic)
       O  extra 'open' repeats at the two fast speeds (the multimodal condition)"""
    rng = random.Random(seed); blocks = []
    b = [dict(mode="free", close_speed=cs, spin=sp, start="half", force=100, hold=1.5, inertia=I, ngrasp=1, rep=r)
         for I in (0.5, 1.0, 2.0) for sp in (-30, -15, 15, 30) for cs in (255, 100) for r in range(2)]
    blocks.append(("F", b))                                                   # 48
    b = [dict(mode="free", close_speed=255, spin=sp, start="half", force=100, hold=0.6, inertia=I, ngrasp=4, rep=r)
         for I in (1.0, 2.0) for sp in (-30, 30) for r in range(3)]
    for row in b: row["ngrasp"] = 3                                          # wrist angle budget
    blocks.append(("R", b))                                                   # 12
    b = [dict(mode="kin", close_speed=cs, spin=sp, start="half", force=F, hold=1.0, inertia="", ngrasp=1, rep=r)
         for F in (40, 100, 180) for cs in (255, 100) for sp in (-24, -12, 0, 12, 24) for r in range(2)]
    blocks.append(("G", b))                                                   # 60
    b = [dict(mode="kin", close_speed=cs, spin=sp, start="half", force=40, hold=0.25, inertia="", ngrasp=1, rep=r)
         for cs in (40, 100, 170, 255) for sp in (-24, 24) for r in range(2)]
    blocks.append(("S", b))                                                   # 16
    b = [dict(mode="kin", close_speed=cs, spin=sp, start="open", force=40, hold=0.25, inertia="", ngrasp=1, rep=r)
         for cs in (170, 255) for sp in (-12, 0, 12) for r in range(4)]
    blocks.append(("O", b))                                                   # 24
    rows = [dict(trial="F1", block="dry", mode="free", close_speed=255, spin=15, start="half", force=100, hold=1.5, inertia=1.0, ngrasp=1, rep=0, split="dry", status=""),
            dict(trial="F2", block="dry", mode="free", close_speed=255, spin=-30, start="half", force=100, hold=1.5, inertia=2.0, ngrasp=1, rep=0, split="dry", status=""),
            dict(trial="G1", block="dry", mode="kin", close_speed=255, spin=12, start="half", force=180, hold=1.0, inertia="", ngrasp=1, rep=0, split="dry", status="")]
    n = len(rows)
    for name, b in blocks:
        rng.shuffle(b)
        for i, row in enumerate(b):
            n += 1; row.update(trial=n, block=name, split="test" if i >= int(0.8 * len(b)) else "calib", status="")
        rows += b
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_COLS); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} trials to {path}  (~{len(rows)*22/60:.0f} min at 22 s each)")

def make_trials3(path, seed=3):
    """Session 2b (kinematic only). H: torque hysteresis sweep w -> -w -> w during a 3 s hold;
    G2: spins +-30 and +-3 at three grip forces; R: three grasp/release cycles."""
    rng = random.Random(seed); blocks = []
    b = [dict(mode="sweep", close_speed=cs, spin=sp, start="half", force=F, hold=3.0, inertia="", ngrasp=1, rep=r)
         for F in (40, 100, 180) for cs in (100, 255) for sp in (12, 30) for r in range(3)]
    blocks.append(("H", b))                                                   # 36
    b = [dict(mode="kin", close_speed=255, spin=sp, start="half", force=F, hold=1.0, inertia="", ngrasp=1, rep=r)
         for F in (40, 100, 180) for sp in (-30, -3, 3, 30) for r in range(2)]
    blocks.append(("G2", b))                                                  # 24
    b = [dict(mode="kin", close_speed=255, spin=sp, start="half", force=100, hold=0.5, inertia="", ngrasp=3, rep=r)
         for sp in (-12, 12) for r in range(3)]
    blocks.append(("R", b))                                                   # 6
    rows = [dict(trial="H1", block="dry", mode="sweep", close_speed=255, spin=30, start="half", force=100, hold=3.0, inertia="", ngrasp=1, rep=0, split="dry", status="")]
    n = 200
    for name, b in blocks:
        rng.shuffle(b)
        for i, row in enumerate(b):
            n += 1; row.update(trial=n, block=name, split="test" if i >= int(0.8 * len(b)) else "calib", status="")
        rows += b
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_COLS); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} trials to {path}  (~{len(rows)*25/60:.0f} min)")

def make_trials4(path, seed=4):
    """Mirrored sweeps: start at -w, go to +w, back to -w, end at +w (1.5 periods over a 4.5 s hold)."""
    rng = random.Random(seed)
    b = [dict(mode="sweep15", close_speed=255, spin=-sp, start="half", force=F, hold=4.5, inertia="", ngrasp=1, rep=r)
         for F in (40, 100, 180) for sp in (12, 30) for r in range(2)]
    rng.shuffle(b); rows = []
    for i, row in enumerate(b):
        row.update(trial=300 + i + 1, block="H2", split="test" if i >= int(0.8 * len(b)) else "calib", status=""); rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_COLS); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} trials to {path}  (~{len(rows)*28/60:.0f} min)")

def make_trials5(path, seed=5):
    """Closed loop on the emulated free spin: regrasp until |w| < W_STOP. loop: fixed 1 s hold;
    looprest: release once the body is at rest in the hand."""
    rng = random.Random(seed); rows = []
    b = [dict(mode="loop", close_speed=255, spin=sp, start="half", force=100, hold=1.0, inertia=I, ngrasp=12, rep=r)
         for I in (0.5, 1.0, 2.0) for sp in (-30, -15, 15, 30) for r in range(2)]
    b += [dict(mode="looprest", close_speed=255, spin=sp, start="half", force=100, hold=1.0, inertia=I, ngrasp=12, rep=0)
          for I in (0.5, 1.0, 2.0) for sp in (-30, -15, 15, 30)]
    rng.shuffle(b)
    rows = [dict(trial="L1", block="dry", mode="loop", close_speed=255, spin=15, start="half", force=100, hold=1.0, inertia=1.0, ngrasp=12, rep=0, split="dry", status=""),
            dict(trial="L2", block="dry", mode="loop", close_speed=255, spin=-30, start="half", force=100, hold=1.0, inertia=2.0, ngrasp=12, rep=0, split="dry", status="")]
    for i, row in enumerate(b):
        row.update(trial=400 + i + 1, block="L" if row["mode"] == "loop" else "LR", split="test" if i >= int(0.8 * len(b)) else "calib", status=""); rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_COLS); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} trials to {path}  (~{len(rows)*35/60:.0f} min)")

def make_trials6(path, seed=6):
    """Release policy on the emulated free spin, single grasp: rigid hand vs hand acting as a rotational damper."""
    rng = random.Random(seed)
    b = [dict(mode=m, close_speed=255, spin=sp, start="half", force=100, hold=2.0, inertia=I, ngrasp=1, rep=r)
         for m in ("rigid", "damp") for I in (0.5, 1.0, 2.0) for sp in (-15, 15) for r in range(2)]
    rng.shuffle(b)
    rows = [dict(trial="P1", block="dry", mode="rigid", close_speed=255, spin=15, start="half", force=100, hold=2.0, inertia=1.0, ngrasp=1, rep=0, split="dry", status=""),
            dict(trial="P2", block="dry", mode="damp", close_speed=255, spin=15, start="half", force=100, hold=2.0, inertia=1.0, ngrasp=1, rep=0, split="dry", status="")]
    for i, row in enumerate(b):
        row.update(trial=500 + i + 1, block="P", split="test" if i >= int(0.8 * len(b)) else "calib", status=""); rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_COLS); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} trials to {path}  (~{len(rows)*30/60:.0f} min)")

def load_trials(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def save_trials(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

# ---------------------------------------------------------------- hardware
class Arm:
    def __init__(self, name, ip):
        self.name, self.ip = name, ip
        print(f"  connecting {name} @ {ip} ...")
        try:                                   # stop a program left running by an earlier session
            from dashboard_client import DashboardClient
            d = DashboardClient(ip); d.connect()
            st = d.programState()
            if "PLAYING" in st or "PAUSED" in st:
                print(f"  {name}: program {st.strip()} on the controller; stopping it"); d.stop(); time.sleep(1.0)
            d.disconnect()
        except Exception as e:
            print(f"  (dashboard check skipped: {e})")
        try:
            self.c = RTDEControl(ip)
            self.r = RTDEReceive(ip, RTDE_HZ, RECORD_VARS)
        except RuntimeError as e:
            if "already in use" not in str(e): raise
            print(f"  primary RTDE registers busy on {name}; using the upper register bank")
            self.c = RTDEControl(ip, -1.0, RTDEControl.FLAG_UPPER_RANGE_REGISTERS)
            self.r = RTDEReceive(ip, RTDE_HZ, RECORD_VARS, False, True)
    def q(self):  return np.asarray(self.r.getActualQ(), float)
    def F(self):  return np.asarray(self.r.getActualTCPForce(), float)
    def stopped(self):
        try:    return self.r.isProtectiveStopped()
        except Exception: return False
    def disconnect(self):
        for f in (self.c.servoStop, self.c.stopScript, self.c.disconnect, self.r.disconnect):
            try: f()
            except Exception: pass

def ensure_control(a, tries=3):
    """Make sure the RTDE control script is running on the controller; reconnect if needed."""
    for i in range(tries):
        try:
            if a.c.isConnected() and a.c.isProgramRunning(): return True
        except Exception: pass
        try:
            if not a.c.isConnected(): a.c.reconnect(); time.sleep(0.5)
            a.c.reuploadScript(); time.sleep(1.0)
            if a.c.isProgramRunning(): return True
        except Exception as e:
            print(f"  {a.name}: control script restart attempt {i+1} failed ({e})")
        time.sleep(1.0)
    print(f"  ! {a.name}: the RTDE control script is NOT running. On the pendant: clear faults / power the arm on, "
          "then press Enter here.")
    input("  >>> Enter to retry ...")
    return ensure_control(a, tries)

def check_arms(*arms):
    """Protective stop: never auto-unlock during contact trials. Ask, then restart the control script."""
    for a in arms:
        if a.stopped():
            print(f"\n  ! {a.name} is PROTECTIVE-STOPPED. Inspect the contact, clear it on the pendant.")
            input("  >>> Enter when cleared ...")
            time.sleep(1.0)
        ensure_control(a)

class TactileLogger:
    def __init__(self, port=None):
        self.s = TactileSensor(port) if port else TactileSensor()
        self.rows, self.on, self.th = [], False, None
    def start(self):
        self.rows, self.on = [], True
        self.th = threading.Thread(target=self._loop, daemon=True); self.th.start()
    def _loop(self):
        while self.on:
            d = self.s.read_single(max_attempts=2)
            if d: self.rows.append([time.time(), d.timestamp] + d.get_forces_array().tolist())
    def stop(self):
        self.on = False
        if self.th: self.th.join(timeout=1.0)
        return self.rows

# ---------------------------------------------------------------- geometry
_AXIS_SIGN = {"v": AXIS_SIGN}
_STRADDLE_POSE = {"v": None}          # chaser TCP pose at the straddle, from the receive interface
def tool_z_axis(ctrl, q):
    """(pose, unit vector along the cone axis pointing from the chaser INTO the cone), base frame.
    Uses the recorded TCP pose of the straddle, not forward kinematics."""
    from scipy.spatial.transform import Rotation
    pose = np.asarray(_STRADDLE_POSE["v"], float)
    return pose, _AXIS_SIGN["v"] * Rotation.from_rotvec(pose[3:]).as_matrix()[:, 2]

def safe_ik(ctrl, pose, qnear):
    if not ctrl.getInverseKinematicsHasSolution(pose, qnear):
        raise RuntimeError(f"pose unreachable: {np.round(pose[:3],3).tolist()}")
    return np.asarray(ctrl.getInverseKinematics(pose, qnear), float)

def joint_path_along_axis(ctrl, q_ref, axis, d_from, d_to, step=0.001):
    """IK along the tool z axis from d_from to d_to (m, relative to the pose of q_ref).
    Returns (distances, list of joint vectors). Aborts on an IK branch jump."""
    pose_ref, _ = tool_z_axis(ctrl, q_ref)
    ds = np.arange(d_from, d_to + 1e-9, step if d_to > d_from else -step)
    qs, qn = [], np.asarray(q_ref, float)
    for d in ds:
        p = pose_ref.copy(); p[:3] = pose_ref[:3] + d * axis
        q = safe_ik(ctrl, p.tolist(), qn.tolist())
        if len(qs) and np.max(np.abs(q[:5] - qs[-1][:5])) > 0.05:
            raise RuntimeError(f"IK jump at d={d*1000:.0f} mm; re-teach the straddle")
        qs.append(q); qn = q
    return ds, qs

# ---------------------------------------------------------------- target F/T bias vs wrist angle
def _basis(th):
    th = np.atleast_1d(th)
    return np.stack([np.ones_like(th), np.sin(th), np.cos(th), np.sin(2*th), np.cos(2*th)], 1)

def bias_model(coef, th):
    """6-vector F/T bias of the target at wrist-3 angle th (rad, actual_q[5])."""
    return (_basis(th) @ np.asarray(coef, float))[0]

def calibrate_bias(A, B, straddle, q_near):
    """Chaser at NEAR (no contact). Turn the target wrist 3 over +-180 deg at 30 deg/s, record the
    F/T reading against the wrist angle and fit a + b sin + c cos (+ 2nd harmonic) per component."""
    qB = np.asarray(straddle["qB"], float); th0 = qB[5]
    print("  calibrating the target F/T bias against the wrist angle (about 40 s, no contact) ...")
    A.c.moveJ(q_near.tolist(), 0.10, 0.2); B.c.moveJ(qB.tolist(), 0.15, 0.3); time.sleep(0.5)
    rate = np.deg2rad(30.0); legs = [np.pi, -2*np.pi, np.pi]           # 0 -> +180 -> -180 -> 0
    th, F = [], []; ang = 0.0
    for leg in legs:
        T = abs(leg) / rate; t_s = time.perf_counter(); a0 = ang
        while True:
            t0 = time.perf_counter(); a = min((t0 - t_s) / T, 1.0); sm = 3*a*a - 2*a*a*a
            ang = a0 + leg * sm; qb = qB.copy(); qb[5] = th0 + ang
            B.c.servoJ(qb.tolist(), 0.5, 0.5, DT, 0.1, 300)
            A.c.servoJ(q_near.tolist(), 0.5, 0.5, DT, 0.1, 300)
            th.append(B.q()[5]); F.append(B.F())
            if a >= 1.0: break
            el = time.perf_counter() - t0
            if el < DT: time.sleep(DT - el)
    A.c.servoStop(); B.c.servoStop(); time.sleep(0.3)
    th = np.array(th); F = np.array(F); X = _basis(th)
    coef, *_ = np.linalg.lstsq(X, F, rcond=None)
    res = F - X @ coef
    from scipy.spatial.transform import Rotation
    zB = Rotation.from_rotvec(np.asarray(B.r.getActualTCPPose(), float)[3:]).as_matrix()[:, 2]
    tz_raw = F[:, 3:] @ zB; tz_res = res[:, 3:] @ zB
    out = dict(coef=coef.tolist(), n=int(len(th)), th_min=float(th.min()), th_max=float(th.max()),
               tau_z_raw_pp=float(tz_raw.max() - tz_raw.min()), tau_z_res_sd=float(tz_res.std()),
               F_res_sd=res.std(0).tolist(), zB=zB.tolist())
    json.dump(out, open(BIAS_FILE, "w"), indent=1)
    print(f"  bias fit: tau_z swing over the sweep {out['tau_z_raw_pp']:.3f} N m peak-to-peak; residual sd {out['tau_z_res_sd']:.3f} N m  -> {BIAS_FILE}")
    return out["coef"]

def calibrate_finger_offset(A, grip, q_act, q_near):
    """Chaser 10 cm back (no contact). Close the fingers slowly from 0 to 255 and record the chaser
    torque about its tool z against the mean finger position: the fingers' own mass shifts the reading."""
    from scipy.spatial.transform import Rotation
    print("  calibrating the chaser torque offset against finger position (about 25 s, no contact) ...")
    A.c.moveJ(q_act.tolist(), 0.10, 0.2); time.sleep(0.3)
    grip._send(0x09, 0, 150, 100); time.sleep(2.0)
    zA = Rotation.from_rotvec(np.asarray(A.r.getActualTCPPose(), float)[3:]).as_matrix()[:, 2]
    F0 = np.mean([A.F() for _ in range(50)], axis=0)
    rows = []
    for target in (255, 0):
        grip._send(0x09, target, 15, 100); t_s = time.time()
        while time.time() - t_s < 14.0:
            st = grip._status(); pos = (st[4] + st[7] + st[10]) / 3.0
            rows.append([pos, float(np.dot(A.F()[3:] - F0[3:], zA))]); time.sleep(0.05)
            if all(c == 3 for c in ((st[1] & 3), (st[1] >> 2) & 3, (st[1] >> 4) & 3)) and time.time() - t_s > 1.0: break
        time.sleep(0.5)
    rows = np.array(rows); edges = np.arange(0, 256, 8); pos_c, tau_c = [], []
    for lo in edges:
        m = (rows[:, 0] >= lo) & (rows[:, 0] < lo + 8)
        if m.sum() >= 2: pos_c.append(float(rows[m, 0].mean())); tau_c.append(float(rows[m, 1].mean()))
    out = dict(pos=pos_c, tau=tau_c, zA=zA.tolist(), n=int(len(rows)), swing=float(max(tau_c) - min(tau_c)))
    json.dump(out, open(FINGER_FILE, "w"), indent=1)
    print(f"  finger offset: torque swing 0..255 = {out['swing']:.3f} N m over {len(pos_c)} bins -> {FINGER_FILE}")
    A.c.moveJ(q_near.tolist(), 0.10, 0.2); time.sleep(0.3)
    return out

def calibrate_hand_offset(A, q_act, q_near, deg=45.0, rate=15.0):
    """Chaser 10 cm back from the rim, fingers half open, no contact. Turn wrist 3 to +deg, -deg and back at `rate`
    deg/s, record the F/T reading against the angle and fit F_j(a) = c0 + c1 sin a + c2 cos a for each component."""
    print(f"  calibrating the chaser F/T offset against its wrist angle (+-{deg:.0f} deg, about {4 * deg / rate:.0f} s, no contact) ...")
    A.c.moveJ(q_act.tolist(), 0.10, 0.2); time.sleep(0.5)
    q0 = np.asarray(q_act, float).copy(); legs = [np.deg2rad(deg), -2 * np.deg2rad(deg), np.deg2rad(deg)]
    ang = 0.0; X, F = [], []
    for leg in legs:
        T = abs(leg) / np.deg2rad(rate); t_s = time.perf_counter(); a0 = ang
        while True:
            t0 = time.perf_counter(); a = min((t0 - t_s) / T, 1.0); sm = 3*a*a - 2*a*a*a
            ang = a0 + leg * sm; q = q0.copy(); q[5] += ang
            A.c.servoJ(q.tolist(), 0.5, 0.5, DT, 0.1, 300)
            X.append(A.q()[5] - q0[5]); F.append(A.F())
            if a >= 1.0: break
            el = time.perf_counter() - t0
            if el < DT: time.sleep(DT - el)
    A.c.servoStop(); time.sleep(0.3)
    X = np.array(X); F = np.array(F); B_ = np.stack([np.ones_like(X), np.sin(X), np.cos(X)], 1)
    coef, *_ = np.linalg.lstsq(B_, F, rcond=None); res = F - B_ @ coef
    from scipy.spatial.transform import Rotation
    swing = (B_ @ coef).max(0) - (B_ @ coef).min(0)
    json.dump(dict(coef=coef.tolist(), deg=deg, n=int(len(X)), swing=swing.tolist(), res_sd=res.std(0).tolist()), open(HAND_OFFSET_FILE, "w"), indent=1)
    print(f"  F/T change over +-{deg:.0f} deg: force {np.round(swing[:3], 2).tolist()} N, torque {np.round(swing[3:], 3).tolist()} N m; "
          f"residual sd {np.round(res.std(0)[3:], 3).tolist()} N m -> {HAND_OFFSET_FILE}")
    A.c.moveJ(q_near.tolist(), 0.10, 0.2); time.sleep(0.3)
    return coef.tolist()

def calibrate_hand_sign(A, B, grip, straddle, qp, force=100):
    """Target held still at the straddle. The chaser approaches along the cone axis on the 1 mm path qp
    (index 0 = straddle, 60 = NEAR), closes on the rim, turns its wrist 3 by +2.5 deg and back, and records the
    torque change. hand_sign = sign(d tau / d theta_hand); a rotation that follows the rim gives the same sign as
    the rim rate in d tau/dt = -K (w_rim - s_h w_hand)."""
    from scipy.spatial.transform import Rotation
    qB = np.asarray(straddle["qB"], float); zA = Rotation.from_rotvec(np.asarray(straddle["poseA"], float)[3:]).as_matrix()[:, 2]
    i_near = int(D_NEAR / 0.001)
    def servo(qa, qb):
        B.c.servoJ(qb.tolist(), 0.5, 0.5, DT, 0.1, 300); A.c.servoJ(np.asarray(qa).tolist(), 0.5, 0.5, DT, 0.1, 300)
    def tauz(): return float(np.dot(A.F()[3:], zA))
    def run_for(T, qa_fn):
        t_s = time.perf_counter()
        while True:
            t0 = time.perf_counter(); a = min((t0 - t_s) / T, 1.0); servo(qa_fn(a), qB)
            if a >= 1.0: break
            el = time.perf_counter() - t0
            if el < DT: time.sleep(DT - el)
    print("  hand-sign calibration: approach, close, turn the chaser wrist +2.5 deg and back (target held still)")
    grip._send(0x09, 66, 150, force); time.sleep(1.0)
    A.c.moveJ(qp[i_near].tolist(), 0.10, 0.2); B.c.moveJ(qB.tolist(), 0.15, 0.3); time.sleep(0.3)
    run_for(3.0, lambda a: qp[int(round(i_near * (1 - (3*a*a - 2*a*a*a))))])
    t_rest = [tauz() for _ in range(50)]; base = float(np.mean(t_rest))
    grip._send(0x09, 255, 100, force); time.sleep(1.2)
    tau0 = float(np.mean([tauz() for _ in range(50)])) - base
    d = np.deg2rad(2.5); q0 = qp[0].copy()
    def rot(x):
        q = q0.copy(); q[5] += x; return q
    run_for(1.0, lambda a: rot(d * a)); time.sleep(0.4)
    tau1 = float(np.mean([tauz() for _ in range(50)])) - base
    run_for(1.0, lambda a: rot(d * (1 - a))); time.sleep(0.3)
    A.c.servoStop(); B.c.servoStop()
    grip._send(0x09, 66, 255, force); time.sleep(1.0)
    run_for(2.0, lambda a: qp[int(round(i_near * a))]); A.c.servoStop()
    dtau = tau1 - tau0
    print(f"  torque after closing {tau0:+.3f} N m, after +2.5 deg {tau1:+.3f} N m, change {dtau:+.3f} N m "
          f"(expected magnitude about {K_GRIP * d:.2f})")
    if abs(dtau) < 0.25:
        sys.exit("  torque change too small to decide the sign; check the grasp and run --hand-sign again")
    sgn = float(np.sign(dtau))
    json.dump(dict(hand_sign=sgn, dtau=dtau, tau0=tau0, tau1=tau1, deg=2.5), open(HAND_SIGN_FILE, "w"), indent=1)
    print(f"  hand_sign = {sgn:+.0f} saved to {HAND_SIGN_FILE}")
    return sgn

# ---------------------------------------------------------------- one trial
def run_trial(A, B, grip, tact, row, half_preset, out_dir, straddle):
    close_speed = int(row["close_speed"])
    w       = np.deg2rad(float(row["spin"]))
    start   = row["start"]
    start_pos = 0 if start == "open" else half_preset
    force   = int(row.get("force") or D.GRIP_FORCE)            # Robotiq rFR
    hold    = float(row.get("hold") or T_GRIP_HOLD)
    mode    = row.get("mode") or "kin"                         # kin: wrist spins at w throughout
    inertia = float(row.get("inertia") or 1.0)                 # free: wrist obeys I*dw/dt = tau_z after contact
    ngrasp  = int(row.get("ngrasp") or 1)
    qB = np.asarray(straddle["qB"], float); qG = np.asarray(straddle["qA"], float)
    ang = {"A": 0.0, "B": 0.0}
    from scipy.spatial.transform import Rotation
    fs = dict(on=False, w=0.0, tau_f=0.0, b=np.zeros(6), bA=np.zeros(3), fbuf=[], fbufA=[], log=[], gc=False)   # free-spin state
    coef = straddle.get("bias_coef")                           # target F/T bias vs wrist angle (None: constant baseline only)
    fo = straddle.get("finger_offset")                         # chaser torque offset vs finger position
    hand_sign = float(straddle.get("hand_sign", 0.0))
    free_like = mode in ("free", "loop", "looprest", "rigid", "damp")
    la_B, gain_B = (SERVO_LA_FREE, SERVO_GAIN_FREE) if free_like else (0.1, 300)
    la_A, gain_A = (SERVO_LA_FREE, SERVO_GAIN_FREE) if mode == "damp" else (0.1, 300)
    if mode == "damp" and straddle.get("hand_offset") is None: raise RuntimeError("damp mode needs hand_offset_A.json")
    if mode == "damp" and hand_sign == 0.0: raise RuntimeError("damp mode needs hand_sign.json: run with --hand-sign first")
    zA = Rotation.from_rotvec(np.asarray(straddle["poseA"], float)[3:]).as_matrix()[:, 2]
    fs["pos"] = float(start_pos); fs["bA6"] = np.zeros(6)
    def finger_offset(pos):
        return float(np.interp(pos, fo["pos"], fo["tau"])) if fo else 0.0
    ho = straddle.get("hand_offset")                           # chaser F/T vs wrist-3 angle (6 x 3 coefficients)
    def hand_off(a):
        if ho is None: return np.zeros(6)
        return np.array([1.0, np.sin(a), np.cos(a)]) @ np.asarray(ho, float) - np.array([1.0, 0.0, 1.0]) @ np.asarray(ho, float)
    def chaser_tau_z():
        """contact torque about the chaser tool z: measured minus settle baseline minus the finger-mass shift minus the
        shift caused by turning the gripper (its weight seen in the rotated sensor frame)"""
        FA = A.F() - hand_off(ang["A"])
        return float(np.dot(FA[3:] - fs["bA6"][3:], zA)) - (finger_offset(fs["pos"]) - finger_offset(start_pos)), FA
    def B_force_corrected():
        F = B.F()
        return F - (bias_model(coef, B.q()[5]) if coef is not None else 0.0)

    def servo_pair(qa_base):
        if abs(ang["B"]) > ANG_MAX: raise RuntimeError("target wrist wound past the limit; trial aborted")
        if abs(ang["B"]) > 0.85 * ANG_MAX: fs["budget"] = True      # closed loop: stop regrasping, finish the trial
        qb = qB.copy(); qb[5] += ang["B"]
        B.c.servoJ(qb.tolist(), 0.5, 0.5, DT, la_B, gain_B)
        qa = np.asarray(qa_base, float).copy(); qa[5] += ang["A"]
        A.c.servoJ(qa.tolist(), 0.5, 0.5, DT, la_A, gain_A)
    def spin(ra, rb):
        """kinematic: the target wrist turns at rb. In free mode after contact, rb is ignored and the
        wrist obeys I dw/dt = tau_z, with tau_z the measured torque about the wrist axis."""
        if fs["on"]:
            tau_raw, FA = chaser_tau_z(); tau = TORQUE_SIGN * tau_raw
            fs["tau_f"] += TAU_ALPHA * (tau - fs["tau_f"])
            fA = float(np.linalg.norm(FA[:3] - fs["bA"]))
            closed = fs.get("closed", False) or (time.perf_counter() - fs.get("t_open", -1e9) < T_GATE_OPEN)
            in_contact = (fA > F_CONTACT or fs["gc"]) and closed   # fingers on the rim, and not opened more than 0.25 s ago
            tq = fs["tau_f"]; tq = 0.0 if abs(tq) < TAU_DB else tq - np.sign(tq) * TAU_DB
            acc = float(np.clip(tq / inertia, -ACC_MAX, ACC_MAX)) if in_contact else 0.0
            fs["w"] = float(np.clip(fs["w"] + acc * DT, -W_MAX, W_MAX))
            fs["log"].append([time.time(), tau, fs["tau_f"], fs["w"], fA, int(in_contact), ang["A"]])
        else:
            fs["w"] = rb
        ang["A"] += ra * DT; ang["B"] += fs["w"] * DT
    def tick(t0):
        el = time.perf_counter() - t0
        if el < DT: time.sleep(DT - el)

    if "path" not in straddle:                    # IK once per session, reused by every trial
        _, axis = tool_z_axis(A.c, qG)
        straddle["path"] = (axis,) + joint_path_along_axis(A.c, qG, axis, -D_NEAR, 0.0)
    axis, d_path, q_path = straddle["path"]
    def q_at(d):
        d = np.clip(d, d_path[0], d_path[-1]); i = int((d - d_path[0]) / 0.001)
        i = min(i, len(q_path) - 2); f = (d - d_path[i]) / 0.001
        return q_path[i] + f * (q_path[i + 1] - q_path[i])

    for arm in (A, B):
        if not (arm.c.isConnected() and arm.c.isProgramRunning()): ensure_control(arm)
    q_start_A = q_at(-D_NEAR)
    if np.max(np.abs(A.q() - q_start_A)) > 0.01 or np.max(np.abs(B.q() - qB)) > 0.01:
        print("    arms not at the start pose; moving there slowly")
        A.c.moveJ(q_start_A.tolist(), 0.10, 0.2); B.c.moveJ(qB.tolist(), 0.15, 0.3); time.sleep(0.3)
        if np.max(np.abs(A.q() - q_start_A)) > 0.01 or np.max(np.abs(B.q() - qB)) > 0.01:
            raise RuntimeError("arms did not reach the start pose (control script not running?)")
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(row=row, t0_host=time.time(), events=[], axis_base=axis.tolist(), half_preset=half_preset,
                params=dict(hand_sign=float(straddle.get("hand_sign", 0.0)), K_GRIP=K_GRIP,
                            servo_B=[SERVO_LA_FREE, SERVO_GAIN_FREE] if mode in ("free", "loop", "looprest", "rigid", "damp") else [0.1, 300], F_RISE=F_RISE, F_RISE_WIN=F_RISE_WIN, TACT_SIGMA=TACT_SIGMA, T_GRIP_HOLD=hold,
                            GRIP_FORCE=force, close_speed=close_speed, start_pos=start_pos, mode=mode, inertia=inertia,
                            ngrasp=ngrasp, TAU_DB=TAU_DB, TAU_ALPHA=TAU_ALPHA, W_MAX=W_MAX, ACC_MAX=ACC_MAX, T_APPROACH=T_APPROACH))
    def ev(name): meta["events"].append([name, time.time()]); print(f"    {name}")

    grip._send(0x09, start_pos, 150, force); time.sleep(1.0)
    A.r.startFileRecording(os.path.join(out_dir, "armA_500Hz.csv"), RECORD_VARS)
    B.r.startFileRecording(os.path.join(out_dir, "armB_500Hz.csv"), RECORD_VARS)
    tact.start(); ev("record_start")

    # 1. spin up the target at NEAR
    t_s = time.perf_counter()
    while True:
        t0 = time.perf_counter(); t = t0 - t_s
        spin(0.0, w * min(t / SPIN_RAMP, 1.0)); servo_pair(q_at(-D_NEAR))
        if t > SPIN_RAMP: break
        tick(t0)
    if not (A.c.isProgramRunning() and B.c.isProgramRunning()):
        raise RuntimeError("RTDE control script dropped during the spin-up")
    if abs(B.r.getActualQd()[5] - w) > 0.3 * abs(w) + 0.02:
        raise RuntimeError(f"target wrist is not spinning ({np.rad2deg(B.r.getActualQd()[5]):.1f} vs {np.rad2deg(w):.1f} deg/s)")
    ev("spinning")

    # 2. NEAR -> straddle along the cone axis, then settle
    t_s = time.perf_counter()
    while True:
        t0 = time.perf_counter(); a = min((t0 - t_s) / T_APPROACH, 1.0); sm = 3*a*a - 2*a*a*a
        spin(0.0, w); servo_pair(q_at(-D_NEAR * (1 - sm)))
        if a >= 1.0: break
        tick(t0)
    t_s = time.perf_counter()
    while time.perf_counter() - t_s < T_SETTLE:
        t0 = time.perf_counter(); spin(0.0, w); servo_pair(q_at(0.0)); fs["fbuf"].append(B_force_corrected()); fs["fbufA"].append(A.F()); tick(t0)
    fs["b"] = np.mean(fs["fbuf"][-25:], axis=0)            # residual target F/T baseline at the straddle, spinning
    fs["bA6"] = np.mean(fs["fbufA"][-25:], axis=0); fs["bA"] = fs["bA6"][:3]
    ev("at_straddle")

    # rest statistics for the online contact test: chaser force (last 0.3 s) and tactile (last 1 s)
    nF = max(int(F_RISE_WIN / DT), 2); fbuf = []
    rows_rest = [r for r in tact.rows if r[0] > time.time() - 1.0]
    if len(rows_rest) >= 10:
        Xr = np.array(rows_rest)[:, 2:11]; t_mu, t_sd = Xr.mean(0), Xr.std(0) + 1.0
    else:
        t_mu = t_sd = None
    meta["tactile_rest_mu"] = None if t_mu is None else t_mu.tolist()
    meta["tactile_rest_sd"] = None if t_sd is None else t_sd.tolist()
    grip_log = []

    # 3. close the fingers on the rim; detect the contact (repeated ngrasp times)
    contact = None; k = 0
    def poll_gripper():
        st = grip._status(); grip_log.append([time.time(), st[1] & 3, (st[1] >> 2) & 3, (st[1] >> 4) & 3, st[4], st[7], st[10]])
        fs["gc"] = any(x in (1, 2) for x in grip_log[-1][1:4]); fs["pos"] = (st[4] + st[7] + st[10]) / 3.0
        return grip_log[-1]
    for g in range(ngrasp):
        grip._send(0x09, 255, close_speed, force); fs["closed"] = True; ev("close_cmd" if g == 0 else f"close_cmd {g+1}")
        t_s = time.perf_counter(); c = None; fbuf = []; last_tact = len(tact.rows)
        while True:
            t0 = time.perf_counter(); spin(0.0, w); servo_pair(q_at(0.0)); k += 1
            f = A.F()[:3]; fbuf.append(f); fbuf = fbuf[-nF:]
            if len(fbuf) == nF and np.linalg.norm(fbuf[-1] - fbuf[0]) > F_RISE:
                c = "force"; meta.setdefault("f_rise_N", float(np.linalg.norm(fbuf[-1] - fbuf[0])))
            if g == 0 and t_mu is not None and len(tact.rows) > last_tact:
                x = np.array(tact.rows[-1][2:11]); dev = np.abs(x - t_mu) / t_sd; last_tact = len(tact.rows)
                if dev.max() > TACT_SIGMA:
                    c = c or "tactile"; meta["tactile_finger"] = int(np.argmax(dev.reshape(3, 3).max(1))) + 1
            if k % 6 == 0:
                gl = poll_gripper()
                if any(x in (1, 2) for x in gl[1:4]): c = c or "finger_code"
            if c:
                ev(f"contact ({c})" if g == 0 else f"contact{g+1} ({c})"); break
            if time.perf_counter() - t_s > T_CLOSE_MAX:
                ev("no_contact_timeout"); break
            if A.stopped() or B.stopped():
                ev("protective_stop"); break
            tick(t0)
        contact = contact or c
        if c and mode in ("free", "loop", "looprest", "rigid", "damp") and not fs["on"]:
            fs["on"] = True; fs["w"] = w; fs["tau_f"] = 0.0; fs["t_first"] = time.perf_counter(); ev(f"free_spin_on w0={np.rad2deg(w):.1f} deg/s I={inertia}")
        t_s = time.perf_counter(); t_rest = None
        def hold_over():
            el = time.perf_counter() - t_s
            if mode == "looprest":                              # release once the body is at rest in the hand
                nonlocal t_rest
                if abs(fs["w"]) < np.deg2rad(1.0):
                    t_rest = t_rest or time.perf_counter()
                    if time.perf_counter() - t_rest > 0.2: return True
                else: t_rest = None
                return el > hold + 1.5
            if mode == "rigid":                                 # hand fixed; release at the end of the hold (not while jamming)
                return (el >= hold and fs["w"] >= -0.02) or el > hold + 1.5
            if mode == "damp":                                  # hand is a rotational damper; release once body and fingers are both unloaded
                calm = abs(fs["tau_f"]) < 0.15 and abs(fs["w"]) < np.deg2rad(1.0)
                return (el >= 1.0 and calm) or el > hold + 1.5
            if mode == "loop":                                  # fixed hold, but never let go while the rim still moves in the jamming direction
                return (el >= hold and fs["w"] >= -0.02) or el > hold + 1.5 or fs.get("budget", False)
            return el >= hold
        while not hold_over():                                   # fingers stay on the rim
            t0 = time.perf_counter()
            # sweep mode: the wrist rate goes w -> -w -> w (cosine) during the hold, two zero crossings
            # sweep: one period over the hold (ends at +w). sweep15: 1.5 periods, so a negative w ends at +|w|
            per = hold if mode == "sweep" else hold / 1.5
            rb = w * np.cos(2 * np.pi * (t0 - t_s) / per) if mode in ("sweep", "sweep15") else w
            ra = 0.0
            if mode == "damp" and fs["on"]:
                # chaser wrist 3 yields under the grasp torque: w_hand = -s_h tau / D, D = sqrt(K I)/2 (critical damping
                # of the stick oscillation tau'' + (K/D) tau' + (K/I) tau = 0)
                D = np.sqrt(K_GRIP * inertia)                   # damping ratio 0.5; loop rate K/D = sqrt(K/I) = 3.5-7 rad/s
                ra_cmd = float(np.clip(-hand_sign * fs["tau_f"] / D, -HAND_RATE_MAX, HAND_RATE_MAX))
                ra = float(np.clip(ra_cmd, fs.get("ra", 0.0) - HAND_ACC_MAX * DT, fs.get("ra", 0.0) + HAND_ACC_MAX * DT))
                if abs(ang["A"] + ra * DT) > HAND_ANG_MAX: ra = 0.0
                fs["ra"] = ra
            spin(ra, rb); servo_pair(q_at(0.0)); k += 1
            if k % 6 == 0: poll_gripper()
            tick(t0)
        if not fs["on"]:                                        # kinematic: stop the target BEFORE opening the fingers
            # Negative spin jams two fingers against the rim (they did not open in trials 238/241). The rim
            # moving in the positive direction frees them (all 36 sweeps released at +w without trouble), so a
            # negative spin is reversed to +W_UNJAM for a moment before the stop.
            rc = fs["w"]                                        # current rate (a sweep may end away from w)
            if rc < -0.01: prof = [(rc, W_UNJAM, 0.4), (W_UNJAM, W_UNJAM, 0.3), (W_UNJAM, 0.0, 0.2)]
            else:          prof = [(rc, 0.0, 0.4)]
            ev("spin_down" if g == 0 else f"spin_down {g+1}")
            for r0, r1, T in prof:
                t_s = time.perf_counter()
                while True:
                    t0 = time.perf_counter(); a_ = min((t0 - t_s) / T, 1.0)
                    spin(0.0, r0 + (r1 - r0) * a_); servo_pair(q_at(0.0)); k += 1
                    if k % 6 == 0: poll_gripper()
                    if a_ >= 1.0: break
                    tick(t0)
        grip._send(0x09, start_pos, 255, force); fs["closed"] = False; fs["t_open"] = time.perf_counter(); ev("release" if g == 0 else f"release {g+1}")
        if fs["on"]:
            pk = max((abs(x[3]) for x in fs["log"]), default=0.0)
            ev(f"w_at_release={np.rad2deg(fs['w']):.1f} deg/s  (peak |w| while held {np.rad2deg(pk):.1f}, hand turned {np.rad2deg(ang['A']):+.1f} deg)")
        t_s = time.perf_counter(); t_open = T_REGRASP if g < ngrasp - 1 else (T_POST_FREE if fs["on"] else 0.8)
        if mode in ("loop", "looprest") and g == ngrasp - 1: t_open = T_POST_FREE
        if mode in ("rigid", "damp"): t_open = 1.5
        resent = False
        while time.perf_counter() - t_s < t_open:               # fingers off; free spin keeps running
            t0 = time.perf_counter(); t = t0 - t_s
            if fs["on"]:
                rb = w
            else:   # kinematic: stop the target within 0.4 s of the release so a finger that is slow to
                    # let go cannot drag the chaser (protective stop); spin up again before a regrasp
                up = min(max((t - (t_open - 0.5)) / 0.5, 0.0), 1.0) if g < ngrasp - 1 else 0.0
                rb = w * up                                     # target is stopped; spin up only before a regrasp
            spin(0.0, rb); servo_pair(q_at(0.0)); k += 1
            if k % 6 == 0:
                gl = poll_gripper()
                if t > 0.3 and not resent and any(x in (1, 2) for x in gl[1:4]):
                    grip._send(0x09, start_pos, 255, force); resent = True; ev("release re-sent (finger still on the rim)")
            tick(t0)
        if not c: break
        if mode in ("loop", "looprest") and fs["on"]:
            if abs(fs["w"]) < W_STOP: ev(f"detumbled after {g+1} grasps, w={np.rad2deg(fs['w']):.1f} deg/s"); break
            if time.perf_counter() - fs["t_first"] > T_LOOP_MAX: ev(f"loop timeout after {g+1} grasps, w={np.rad2deg(fs['w']):.1f} deg/s"); break
            if fs.get("budget"): ev(f"wrist angle budget used after {g+1} grasps, w={np.rad2deg(fs['w']):.1f} deg/s"); break
    if fs["on"]:
        ev(f"w_end={np.rad2deg(fs['w']):.1f} deg/s"); meta["free_log"] = fs["log"]; meta["free_log_cols"] = ["t", "tau_z", "tau_filt", "w", "fA", "in_contact", "hand_angle"]
        meta["free_bias_FT"] = fs["b"].tolist(); meta["bias_coef"] = coef; meta["finger_offset"] = fo; meta["torque_source"] = "chaser tool z"
        w_end = fs["w"]; fs["on"] = False
    else:
        w_end = 0.0                                             # target already stopped at the release
    meta["gripper_log"] = grip_log; meta["gripper_log_cols"] = ["t", "gDTA", "gDTB", "gDTC", "posA", "posB", "posC"]
    d_hold = 0.0; contact = bool(contact)

    # 5. retract along the cone axis on the precomputed path, spin down, unwind
    d_end = max(d_hold - D_RETRACT, -D_NEAR)
    t_s = time.perf_counter()
    while True:
        t0 = time.perf_counter(); s_ = min((time.perf_counter() - t_s) * V_RETRACT / (d_hold - d_end), 1.0)
        spin(0.0, w_end); servo_pair(q_at(d_hold + s_ * (d_end - d_hold)))
        if s_ >= 1.0: break
        tick(t0)
    ev("retracted")
    q_last = q_at(d_end)
    t_s = time.perf_counter()
    while True:                                   # spin down over 0.6 s
        t0 = time.perf_counter(); f = max(1.0 - (t0 - t_s) / 0.6, 0.0)
        spin(0.0, w_end * f); servo_pair(q_last)
        if f <= 0.0: break
        tick(t0)
    t_s = time.perf_counter(); a0, b0 = ang["A"], ang["B"]; T_UN = max(1.0, abs(a0)/0.6, abs(b0)/0.6)
    while True:                                   # unwind both wrists to their base angle
        t0 = time.perf_counter(); a = min((t0 - t_s) / T_UN, 1.0); s = 3*a*a - 2*a*a*a
        ang["A"] = a0 * (1 - s); ang["B"] = b0 * (1 - s); servo_pair(q_last)
        if a >= 1.0: break
        tick(t0)
    A.c.servoStop(); B.c.servoStop(); time.sleep(0.2)
    A.r.stopFileRecording(); B.r.stopFileRecording(); rows = tact.stop(); ev("record_stop")
    json.dump(rows, open(os.path.join(out_dir, "tactile.json"), "w"))
    meta["contact"] = contact; meta["t1_host"] = time.time()
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    check_arms(A, B)
    return contact                                   # arms are left at NEAR, ready for the next trial

# ---------------------------------------------------------------- preset finder
def find_half_preset(grip, A, B, straddle):
    """At the taught straddle (fingers around the rim), close slowly until a finger stops;
    the half preset is 60 % of the way from open to that value."""
    qA = np.asarray(straddle["qA"], float); qB = np.asarray(straddle["qB"], float)
    if np.max(np.abs(A.q() - qA)) > 0.02 or np.max(np.abs(B.q() - qB)) > 0.02:
        print("  arms are not at the taught straddle; moving there slowly ...")
        B.c.moveJ(qB.tolist(), 0.15, 0.3); A.c.moveJ(qA.tolist(), 0.15, 0.3)
    grip.open(); time.sleep(0.5)
    def finger_codes():
        s = grip._status()                      # byte 1: gDTA bits 0-1, gDTB bits 2-3, gDTC bits 4-5
        return [(s[1] & 0x03), (s[1] >> 2) & 0x03, (s[1] >> 4) & 0x03]   # 0 moving, 1/2 contact, 3 at position
    for pos in range(10, 256, 5):
        grip._send(0x09, pos, 60, D.GRIP_FORCE)
        t0 = time.time(); codes = [0, 0, 0]
        while time.time() - t0 < 1.5:          # wait until finger A has stopped (any reason)
            time.sleep(0.05); codes = finger_codes()
            if codes[0] != 0: break
        if any(c in (1, 2) for c in codes):     # stopped by contact, not by reaching the position
            print(f"  finger contact at value {pos}  (finger codes {codes})")
            half = int(0.6 * pos); grip.open()
            json.dump(dict(contact_value=pos, half=half), open(PRESET_FILE, "w"))
            print(f"  half preset = {half}  saved to {PRESET_FILE}")
            return half
    grip.open(); print("  no finger contact found; keeping default"); return HALF_PRESET_DEFAULT

def teach_straddle():
    """Record the current joints of both arms as the straddle. Nothing moves."""
    ra = RTDEReceive(D.GRIPPER_ROBOT_IP); rb = RTDEReceive(D.CUBESAT_ROBOT_IP)
    st = dict(qA=[round(v, 5) for v in ra.getActualQ()], qB=[round(v, 5) for v in rb.getActualQ()],
              poseA=[round(v, 5) for v in ra.getActualTCPPose()], poseB=[round(v, 5) for v in rb.getActualTCPPose()],
              taught=time.strftime("%Y-%m-%d %H:%M"))
    ra.disconnect(); rb.disconnect()
    json.dump(st, open(STRADDLE_FILE, "w"), indent=1)
    print(f"saved {STRADDLE_FILE}\n  chaser qA = {st['qA']}\n  target qB = {st['qB']}")

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="?"); ap.add_argument("--make", metavar="CSV")
    ap.add_argument("--find-preset", action="store_true"); ap.add_argument("--teach-straddle", action="store_true")
    ap.add_argument("--reps", type=int, default=2); ap.add_argument("--make2", metavar="CSV"); ap.add_argument("--recalib", action="store_true"); ap.add_argument("--make3", metavar="CSV"); ap.add_argument("--make4", metavar="CSV"); ap.add_argument("--make5", metavar="CSV"); ap.add_argument("--make6", metavar="CSV"); ap.add_argument("--hand-sign", action="store_true")
    args = ap.parse_args()
    if args.make: make_trials(args.make, reps=args.reps); return
    if args.make2: make_trials2(args.make2); return
    if args.make3: make_trials3(args.make3); return
    if args.make4: make_trials4(args.make4); return
    if args.make5: make_trials5(args.make5); return
    if args.make6: make_trials6(args.make6); return
    if args.teach_straddle: teach_straddle(); return
    if not os.path.exists(STRADDLE_FILE): sys.exit("no straddle.json: put the fingers around the rim by hand, then run --teach-straddle")
    straddle = json.load(open(STRADDLE_FILE))
    if "axis_sign" in straddle: _AXIS_SIGN["v"] = straddle["axis_sign"]
    _STRADDLE_POSE["v"] = straddle["poseA"]
    if not args.trials: ap.print_help(); return
    rows = load_trials(args.trials)
    half = json.load(open(PRESET_FILE))["half"] if os.path.exists(PRESET_FILE) else HALF_PRESET_DEFAULT

    A = B = grip = tact = None
    try:
        print("Connecting ...")
        A = Arm("gripper-arm", D.GRIPPER_ROBOT_IP); B = Arm("cubesat-arm", D.CUBESAT_ROBOT_IP)
        grip = D.Gripper3F(D.GRIPPER_IP); grip.connect()
        tact = TactileLogger()
        print("Keep both e-stops in hand.\n")
        if args.find_preset and D.confirm("find the half-closed finger preset at the straddle"):
            check_arms(A, B)
            # --- which way is "away from the cone"? probe 1 cm, ask, remember ---
            qA = np.asarray(straddle["qA"], float)
            if "axis_sign" in straddle:
                _AXIS_SIGN["v"] = straddle["axis_sign"]
            else:
                pose0, _ = tool_z_axis(A.c, qA)
                try:
                    print("  recorded TCP pose :", np.round(pose0, 4).tolist())
                    print("  live TCP pose     :", np.round(A.r.getActualTCPPose(), 4).tolist())
                    print("  FK of qA          :", np.round(A.c.getForwardKinematics(qA.tolist()), 4).tolist())
                    print("  TCP offset        :", np.round(A.c.getTCPOffset(), 4).tolist())
                except Exception as e:
                    print("  (diagnostic failed:", e, ")")
                from scipy.spatial.transform import Rotation
                z = Rotation.from_rotvec(pose0[3:]).as_matrix()[:, 2]
                reach = {}
                for sgn in (+1, -1):
                    p = pose0.copy(); p[:3] = pose0[:3] - sgn * z * D_ACTIVATE   # "back" = -axis
                    reach[sgn] = bool(A.c.getInverseKinematicsHasSolution(p.tolist(), qA.tolist()))
                print(f"  reachable 10 cm back:  sign +1: {reach[+1]}   sign -1: {reach[-1]}")
                sgn = +1 if reach[+1] and not reach[-1] else (-1 if reach[-1] and not reach[+1] else +1)
                _AXIS_SIGN["v"] = sgn
                if D.confirm(f"probe: move the chaser 1 cm 'back' with sign {sgn:+d} (watch it)"):
                    p = pose0.copy(); p[:3] = pose0[:3] - sgn * z * 0.01
                    A.c.moveJ(safe_ik(A.c, p.tolist(), qA.tolist()).tolist(), 0.05, 0.1); time.sleep(0.3)
                    ans = input("  did it move AWAY from the cone? [y/n]: ").strip().lower()
                    A.c.moveJ(qA.tolist(), 0.05, 0.1); time.sleep(0.3)
                    if ans != "y":
                        sgn = -sgn; _AXIS_SIGN["v"] = sgn
                        p = pose0.copy(); p[:3] = pose0[:3] - sgn * z * D_ACTIVATE
                        if not A.c.getInverseKinematicsHasSolution(p.tolist(), qA.tolist()):
                            sys.exit("  the away-from-cone direction is not reachable 10 cm back; re-teach the straddle with the arm less stretched")
                straddle["axis_sign"] = sgn
                json.dump({k: v for k, v in straddle.items() if k != "path"}, open(STRADDLE_FILE, "w"), indent=1)
                print(f"  axis sign {sgn:+d} saved to {STRADDLE_FILE}")
            if grip.is_active():
                print("  gripper already active: no finger sweep")
            else:
                # activation sweeps the fingers fully closed and open: never do it at the rim
                _, axis = tool_z_axis(A.c, qA)
                _, qp = joint_path_along_axis(A.c, qA, axis, 0.0, -D_ACTIVATE)
                print(f"  gripper not active: backing the chaser {D_ACTIVATE*100:.0f} cm off the cone before the activation sweep")
                if not D.confirm(f"move chaser {D_ACTIVATE*100:.0f} cm back, activate the gripper there, and return"): return
                A.c.moveJ(qp[-1].tolist(), 0.10, 0.2); time.sleep(0.3)
                grip.activate(); grip.open(); time.sleep(0.5)
                A.c.moveJ(qA.tolist(), 0.10, 0.2); time.sleep(0.3)
            half = find_half_preset(grip, A, B, straddle)
            print(f"  half preset = {half}  (done; run again without --find-preset for the trials)"); return
        print(f"  half preset = {half}")
        if "axis_sign" not in straddle: sys.exit("run --find-preset first (it finds the approach direction)")
        qA = np.asarray(straddle["qA"], float)
        _, axis = tool_z_axis(A.c, qA)
        _, qp = joint_path_along_axis(A.c, qA, axis, 0.0, -D_ACTIVATE)   # 1 mm steps, 0 .. -100 mm
        q_act, q_near = qp[-1], qp[int(D_NEAR / 0.001)]
        if D.confirm("move target to the straddle and chaser to NEAR (60 mm back along the cone axis)"):
            check_arms(A, B)
            B.c.moveJ(straddle["qB"], 0.15, 0.3)
            if not grip.is_active():
                print("  gripper not active: backing off 10 cm for the activation sweep")
                A.c.moveJ(q_act.tolist(), 0.10, 0.2); time.sleep(0.3)
                grip.activate(); grip.open(); time.sleep(0.5)
            A.c.moveJ(q_near.tolist(), 0.10, 0.2); time.sleep(0.3)
            print("  chaser at NEAR (6 cm back from the straddle)")
        if os.path.exists(HAND_SIGN_FILE) and not args.hand_sign:
            straddle["hand_sign"] = json.load(open(HAND_SIGN_FILE))["hand_sign"]; print(f"  hand sign {straddle['hand_sign']:+.0f} loaded from {HAND_SIGN_FILE}")
        elif any((r.get("mode") or "kin") == "damp" and r["status"] != "done" for r in rows) or args.hand_sign:
            if D.confirm("calibrate the hand sign (approach, close on the still rim, turn the chaser wrist 2.5 deg)"):
                check_arms(A, B); straddle["hand_sign"] = calibrate_hand_sign(A, B, grip, straddle, qp); check_arms(A, B)
                A.c.moveJ(q_near.tolist(), 0.10, 0.2); B.c.moveJ(straddle["qB"], 0.15, 0.3)
        if any((r.get("mode") or "kin") == "damp" and r["status"] != "done" for r in rows):
            if os.path.exists(HAND_OFFSET_FILE) and not args.recalib:
                straddle["hand_offset"] = json.load(open(HAND_OFFSET_FILE))["coef"]; print(f"  hand offset model loaded from {HAND_OFFSET_FILE}")
            elif D.confirm("calibrate the chaser F/T offset vs its wrist angle (10 cm back, gripper turns +-45 deg, no contact)"):
                check_arms(A, B); straddle["hand_offset"] = calibrate_hand_offset(A, q_act, q_near); check_arms(A, B)
        if any((r.get("mode") or "kin") in ("free", "loop", "looprest", "rigid", "damp") and r["status"] != "done" for r in rows):
            if os.path.exists(BIAS_FILE):
                straddle["bias_coef"] = json.load(open(BIAS_FILE))["coef"]
            if os.path.exists(FINGER_FILE) and not args.recalib:
                straddle["finger_offset"] = json.load(open(FINGER_FILE)); print(f"  finger offset model loaded from {FINGER_FILE}")
            elif D.confirm("calibrate the chaser torque offset vs finger position (10 cm back, fingers sweep, no contact)"):
                check_arms(A, B); straddle["finger_offset"] = calibrate_finger_offset(A, grip, q_act, q_near); check_arms(A, B)

        todo = [r for r in rows if r["status"] not in ("done", "skipped")]
        print(f"\n{len(todo)} trials to run, {len(rows)-len(todo)} already done.")
        for r in todo:
            print(f"\n--- trial {r['trial']}: mode={r.get('mode') or 'kin'}  close speed={r['close_speed']}  spin={r['spin']} deg/s  "
                  f"start={r['start']}  force={r.get('force') or D.GRIP_FORCE}  hold={r.get('hold') or T_GRIP_HOLD}  "
                  f"I={r.get('inertia') or '-'}  grasps={r.get('ngrasp') or 1}  [{r['split']}]")
            if r is todo[0] and not D.confirm("start trials (Enter) "): break
            tid = r['trial']; out = os.path.join(OUT_ROOT, f"trial_{int(tid):04d}" if tid.isdigit() else f"trial_{tid}")
            try:
                ok = run_trial(A, B, grip, tact, r, half, out, straddle)
                r["status"] = "done" if ok else "no_contact"
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"  ! trial failed: {e}"); r["status"] = "failed"
                try: grip._send(0x09, 0, 255, D.GRIP_FORCE)      # let go of the cone first
                except Exception: pass
                try: A.c.servoStop(); B.c.servoStop(); A.r.stopFileRecording(); B.r.stopFileRecording(); tact.stop()
                except Exception: pass
                check_arms(A, B)
                try:                                              # back to the start pose slowly
                    time.sleep(1.0); A.c.moveJ(q_near.tolist(), 0.10, 0.2); B.c.moveJ(straddle["qB"], 0.15, 0.3)
                except Exception as e2: print(f"  ! could not return to NEAR: {e2}")
                if input("  continue with next trial? [y/N]: ").strip().lower() != "y": break
            save_trials(args.trials, rows)
        print("\nDone for this session.")
    except KeyboardInterrupt:
        print("\nInterrupted."); save_trials(args.trials, rows)
    finally:
        for x in (grip, A, B):
            try: x.disconnect()
            except Exception: pass
        try: tact.s.disconnect()
        except Exception: pass

if __name__ == "__main__":
    main()
