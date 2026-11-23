# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
import time
GRIPPER_ROBOT_IP="a"; CUBESAT_ROBOT_IP="b"; GRIPPER_IP="g"; GRIP_FORCE=40; DT=1/125
def confirm(m): print("confirm:",m); return True
class Gripper3F:
    def __init__(self,ip): self.t=None
    def connect(self): pass
    def is_active(self): return True
    def activate(self): pass
    def open(self): pass
    def _send(self,cmd,pos,spd,frc):
        import builtins; builtins.MOCK_GRIP = (pos==255); self.t=time.time(); self.pos=pos; print(f"      gripper -> pos {pos} speed {spd} force {frc}")
    def _status(self):
        st=[0]*16
        if self.t and self.pos==255 and time.time()-self.t>0.5: st[1]=0b000100
        return st
    def disconnect(self): pass
