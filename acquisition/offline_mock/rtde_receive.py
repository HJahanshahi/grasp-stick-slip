# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
import numpy as np, time, builtins, json
class RTDEReceiveInterface:
    """stick-slip mock: tau = clip(-K (theta_body - s*theta_hand)), body angle from MOCK_W integration"""
    def __init__(self,*a,**k): self.t0=time.time(); self.th=0.0; self.tl=time.time()
    def getActualQ(self):
        d=json.load(open('straddle.json')); return d['qA']
    def getActualQd(self): return [0,0,0,0,0,getattr(builtins,'MOCK_SPIN',0.2618)]
    def getActualTCPForce(self):
        now=time.time(); dt=now-self.tl; self.tl=now
        w=getattr(builtins,'MOCK_W',None); h=getattr(builtins,'MOCK_HAND',0.0)
        grip=getattr(builtins,'MOCK_GRIP',False)
        if w is None: w = 0.0
        if not grip: tau=0.0; self.th=0.0
        else:
            self.th += w*dt
            tau=float(np.clip(-24.3*(self.th - (-1.0)*h), -1.0, 3.3))
            # slip: re-anchor
            self.th = -tau/24.3 + (-1.0)*h
        from scipy.spatial.transform import Rotation
        zA=Rotation.from_rotvec(np.array(json.load(open('straddle.json'))['poseA'][3:])).as_matrix()[:,2]
        f=np.random.randn(6)*0.002; f[2]+= 40.0 if grip else 0.0; f[3:]+=tau*zA
        return list(f)
    def getActualTCPPose(self): return [0,0,0,0,0,0]
    def isProtectiveStopped(self): return False
    def startFileRecording(self,*a): pass
    def stopFileRecording(self): pass
    def disconnect(self): pass
