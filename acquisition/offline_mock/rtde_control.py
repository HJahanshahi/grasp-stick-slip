# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
import numpy as np, builtins
class RTDEControlInterface:
    FLAG_UPPER_RANGE_REGISTERS=1
    def __init__(self,ip,*a,**k): self.ip=ip; self.q0=None
    def servoJ(self,q,*a):
        if self.ip=="a":
            if self.q0 is None: self.q0=q[5]
            builtins.MOCK_HAND=q[5]-self.q0
    def servoStop(self): pass
    def stopScript(self): pass
    def disconnect(self): pass
    def isProgramRunning(self): return True
    def isConnected(self): return True
    def reconnect(self): pass
    def reuploadScript(self): pass
    def getInverseKinematicsHasSolution(self,p,q): return True
    def getInverseKinematics(self,p,q): return list(np.asarray(q))
    def moveJ(self,*a): pass
