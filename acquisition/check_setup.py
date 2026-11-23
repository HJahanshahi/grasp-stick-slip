# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""
check_setup.py  -  one-time check of the testbed before the impact experiments.

Run on the lab computer:   python check_setup.py
Nothing moves. It only reads.

Checks
  1. ur_rtde version and whether 500 Hz file recording is available
  2. Both robots: PolyScope version, robot mode, safety mode, current safety status
  3. RTDE receive rate actually achieved from each robot (target 500 Hz)
  4. Tactile sensor: sample rate and value ranges (needs tactile_sensor.py in the same folder)

Safety limits (tool force, power, speed) CANNOT be written from RTDE or the dashboard.
They live in the password-protected safety configuration on the pendant:
  Installation > Safety > Robot Limits  (needs the safety password, then "Apply").
This script only reads the current safety mode so we know what we are running under.
"""
import sys, time, importlib.metadata as md

MOCKUP_IP  = "192.168.1.5"
GRIPPER_IP = "192.168.1.4"
TACTILE_PORT = "/dev/ttyUSB1"
RTDE_HZ = 500.0

def hr(t): print("\n" + "=" * 64 + f"\n {t}\n" + "=" * 64)

# ---------------------------------------------------------------- 1. ur_rtde
hr("1. ur_rtde")
try:
    print("ur_rtde version:", md.version("ur_rtde"))
except Exception as e:
    print("ur_rtde version: unknown (", e, ")")
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface as RTDEControl
print("startFileRecording available:", hasattr(RTDEReceive, "startFileRecording"))
print("RTDEReceive accepts a frequency argument:",
      "frequency" in RTDEReceive.__init__.__doc__ if RTDEReceive.__init__.__doc__ else "unknown")

# ---------------------------------------------------------------- 2. robots
try:
    from dashboard_client import DashboardClient
except ImportError:
    DashboardClient = None
    print("dashboard_client not importable; skipping version/safety read")

for name, ip in [("mock-up arm", MOCKUP_IP), ("gripper arm", GRIPPER_IP)]:
    hr(f"2. {name}  {ip}")
    if DashboardClient:
        try:
            d = DashboardClient(ip); d.connect()
            print("PolyScope:     ", d.polyscopeVersion())
            print("Robot mode:    ", d.robotmode())
            print("Safety status: ", d.safetystatus())
            d.disconnect()
        except Exception as e:
            print("dashboard read failed:", e)
    # ------------------------------------------------------------ 3. rate
    try:
        r = RTDEReceive(ip, RTDE_HZ)
        print("RTDE safety mode code:", r.getSafetyMode(), " robot mode code:", r.getRobotMode())
        print("TCP force now (N, Nm):", [round(x, 2) for x in r.getActualTCPForce()])
        n = 0; last = None; t0 = time.time()
        while time.time() - t0 < 3.0:
            ts = r.getTimestamp()
            if ts != last:
                n += 1; last = ts
        print(f"distinct samples in 3 s: {n}  ->  {n/3.0:.0f} Hz achieved (target {RTDE_HZ:.0f})")
        r.disconnect()
    except Exception as e:
        print("RTDE connect/rate test failed:", e)

# ---------------------------------------------------------------- 4. tactile
hr("4. tactile sensor")
try:
    from tactile_sensor import TactileSensor
    s = TactileSensor(TACTILE_PORT)
    n = 0; vals = []; t0 = time.time()
    while time.time() - t0 < 5.0:
        d = s.read_single()
        if d:
            n += 1; vals.append(d.get_forces_array())
    import numpy as np
    vals = np.array(vals)
    print(f"samples in 5 s: {n}  ->  {n/5.0:.0f} Hz")
    if len(vals):
        print("channels:", vals.shape[1])
        print("min per channel:", vals.min(0).round(0).tolist())
        print("max per channel:", vals.max(0).round(0).tolist())
        print("std per channel:", vals.std(0).round(1).tolist())
    print("class file:", sys.modules["tactile_sensor"].__file__)
except Exception as e:
    print("tactile test skipped:", e)

print("\nDone.")
