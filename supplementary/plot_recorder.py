# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""plot_recorder.py - records every plotted series (lines, markers) of each figure when it is saved, as JSON in the
folder named by the environment variable DUMP_DIR. Imported by make_data_S1.py before a figure script runs."""
import os, json, numpy as np, matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.colors import to_hex
D=os.environ.get('DUMP_DIR','.'); _orig=Figure.savefig; _done=set()
def L(a): 
    a=np.ma.filled(np.ma.masked_invalid(np.asarray(a,float)),np.nan); return [None if np.isnan(v) else float(v) for v in a.ravel()]
def _col(l):
    if l.get_linestyle() not in('None','',' '): return to_hex(l.get_color())
    f=str(l.get_markerfacecolor()); e=to_hex(l.get_markeredgecolor())
    if f=='none' or to_hex(l.get_markerfacecolor())=='#ffffff': return e+' open'
    return to_hex(l.get_markerfacecolor())
def dump(fig,name):
    out=[]
    for i,ax in enumerate(fig.axes):
        A=dict(axis=i,xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),title=ax.get_title(),pos=[round(v,3) for v in ax.get_position().bounds],series=[])
        for j,l in enumerate(ax.lines):
            x,y=l.get_data()
            try: x=np.asarray(x,float); y=np.asarray(y,float)
            except Exception: continue
            if len(x)==0: continue
            A['series'].append(dict(kind='line' if l.get_linestyle() not in('None','',' ') else 'markers',label=l.get_label(),color=_col(l),ls=str(l.get_linestyle()),marker=str(l.get_marker()),x=L(x),y=L(y)))
        for j,c in enumerate(ax.collections):
            try: o=np.asarray(c.get_offsets(),float)
            except Exception: continue
            if o.ndim==2 and len(o) and type(c).__name__=='PathCollection':
                fc=c.get_facecolors(); ecol=c.get_edgecolors(); col=(to_hex(fc[0]) if len(fc) and fc[0][3]>0 else (to_hex(ecol[0])+' open' if len(ecol) else ''))
                A['series'].append(dict(kind='scatter',label=c.get_label(),color=col,ls='',marker=str(len(c.get_paths()[0].vertices)),x=L(o[:,0]),y=L(o[:,1])))
        bars=[p for p in ax.patches if type(p).__name__=='Rectangle']
        if bars: A['series'].append(dict(kind='bars',label='bars',color='',ls='',marker='',x=L([p.get_x()+p.get_width()/2 for p in bars]),y=L([p.get_y()+p.get_height() for p in bars])))
        for cont in ax.containers:
            if type(cont).__name__=='ErrorbarContainer' and cont.lines[2]:
                for lc in cont.lines[2]:
                    segs=lc.get_segments(); A['series'].append(dict(kind='errorbar',label=str(cont.get_label()),color='',ls='',marker='',x=L([s[0][0] for s in segs]+[s[1][0] for s in segs]),y=L([s[0][1] for s in segs]+[s[1][1] for s in segs])))
        out.append(A)
    json.dump(out,open(os.path.join(D,name+'.json'),'w'))
def savefig(self,fname,*a,**k):
    n=os.path.splitext(os.path.basename(str(fname)))[0]
    if n not in _done: _done.add(n); dump(self,n)
    return _orig(self,fname,*a,**k)
Figure.savefig=savefig
