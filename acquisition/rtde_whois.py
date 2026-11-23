# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""
rtde_whois.py  -  find out which RTDE input registers are in use on each robot, and by whom.

Speaks the RTDE protocol directly (port 30004) and asks the controller to set up an input
recipe for every input variable, one at a time. The controller answers with the variable
type, or IN_USE if another RTDE client already owns it. Nothing is changed on the robot.
Also lists this computer's TCP connections to the robots and the network interfaces,
to spot a ROS driver or a VM.

  python3 rtde_whois.py
"""
import socket, struct, subprocess, sys

ROBOTS = {"mock-up arm": "192.168.1.5", "gripper arm": "192.168.1.4"}
PORT = 30004
INPUT_VARS = (["speed_slider_mask", "speed_slider_fraction",
               "standard_digital_output_mask", "standard_digital_output",
               "configurable_digital_output_mask", "configurable_digital_output",
               "tool_digital_output_mask", "tool_digital_output",
               "standard_analog_output_mask", "standard_analog_output_type",
               "standard_analog_output_0", "standard_analog_output_1",
               "input_bit_registers0_to_31", "input_bit_registers32_to_63", "external_force_torque"]
              + [f"input_bit_register_{i}" for i in range(64, 128)]
              + [f"input_int_register_{i}" for i in range(48)]
              + [f"input_double_register_{i}" for i in range(48)])

def rtde_cmd(sock, cmd, payload=b""):
    sock.sendall(struct.pack(">HB", 3 + len(payload), cmd) + payload)
    hdr = b""
    while len(hdr) < 3:
        hdr += sock.recv(3 - len(hdr))
    size, typ = struct.unpack(">HB", hdr)
    body = b""
    while len(body) < size - 3:
        body += sock.recv(size - 3 - len(body))
    while typ != cmd:                      # skip unsolicited text messages
        hdr = b""
        while len(hdr) < 3:
            hdr += sock.recv(3 - len(hdr))
        size, typ = struct.unpack(">HB", hdr)
        body = b""
        while len(body) < size - 3:
            body += sock.recv(size - 3 - len(body))
    return body

def probe(ip):
    s = socket.create_connection((ip, PORT), timeout=3)
    ok = rtde_cmd(s, 86, struct.pack(">H", 2))             # request protocol version 2
    if not ok or ok[0] != 1:
        print("  protocol v2 refused"); s.close(); return
    v = rtde_cmd(s, 118)                                    # controller version
    print("  controller version:", ".".join(str(x) for x in struct.unpack(">IIII", v)))
    in_use, other = [], {}
    for name in INPUT_VARS:
        body = rtde_cmd(s, 73, name.encode())               # setup inputs, one variable
        rid, types = body[0], body[1:].decode(errors="ignore")
        if "IN_USE" in types:
            in_use.append(name)
        elif rid == 0:
            other[name] = types
    s.close()
    if in_use:
        print(f"  IN USE by another RTDE client ({len(in_use)}):")
        groups = {}
        for n in in_use:
            key = n.rsplit("_", 1)[0] if n[-1].isdigit() else n
            groups.setdefault(key, []).append(n.rsplit("_", 1)[-1] if n[-1].isdigit() else "")
        for k, v in groups.items():
            print(f"    {k}" + (f"  [{', '.join(v)}]" if any(v) else ""))
    else:
        print("  no input register is in use: the robot side is free")
    if other:
        print("  not available:", ", ".join(f"{k} ({v})" for k, v in list(other.items())[:5]), "...")

def local_connections():
    print("\nTCP connections from this computer to the robots (ports 30001-30004, 50002, 29999):")
    try:
        out = subprocess.run(["ss", "-tnp"], capture_output=True, text=True).stdout
        hits = [l for l in out.splitlines() if any(f":{p}" in l for p in (30001, 30002, 30003, 30004, 50002, 29999))]
        print("\n".join("  " + h for h in hits) if hits else "  none")
    except Exception as e:
        print("  (ss failed:", e, ")")
    print("\nNetwork interfaces on this computer:")
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True).stdout
        for l in out.splitlines():
            print("  " + " ".join(l.split()[1:4]))
    except Exception as e:
        print("  (ip failed:", e, ")")
    print("\nVirtual machines / containers that could run a ROS driver:")
    for cmd in (["VBoxManage", "list", "runningvms"], ["docker", "ps", "--format", "{{.Names}} {{.Image}}"],
                ["pgrep", "-a", "-f", "ur_robot_driver|urcl|ros2|roscore|rosmaster"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
            print(f"  {cmd[0]}: {out if out else 'none'}")
        except Exception:
            print(f"  {cmd[0]}: not installed")

if __name__ == "__main__":
    for name, ip in ROBOTS.items():
        print(f"\n=== {name}  {ip}")
        try:
            probe(ip)
        except Exception as e:
            print("  probe failed:", e)
    local_connections()
