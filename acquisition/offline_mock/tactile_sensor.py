# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
import numpy as np, time
class _D:
    def __init__(self): self.timestamp=time.time()
    def get_forces_array(self): return np.random.randn(9)
class TactileSensor:
    def __init__(self,*a): pass
    def read_single(self,max_attempts=2): time.sleep(0.02); return _D()
    def disconnect(self): pass
