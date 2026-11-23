# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""figS4_grasp_loss.py - fig. S4: torque against rim angle while the rigid hand holds the body (block P, one-axis
emulation), and the energy loss of the grasp per half swing, measured and from the law driven by the same motion."""
import numpy as np, json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "..", "figures"); os.makedirs(FIG, exist_ok=True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Inter", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 600, "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5, "axes.axisbelow": True})
z=np.load(sys.argv[1] if len(sys.argv)>1 else os.path.join(HERE,"data","block_P_pack.npz"),allow_pickle=True); CO="#D55E00"; trs=sorted({k.split('/')[0] for k in z.files})
Pp=np.array([18.226839128435657,7.576894038658541,2.5894049765172324,7.557897412443152e-27,0.5641941560541658,4.4045840701688315,1.8077901051997332,1.434108483334086,40.0]); k,cp,cn=Pp[:3],Pp[3:6],Pp[6:9]
def law(tt,wr,F):
    e=np.zeros(3); out=[]; V=[]
    for i in range(len(tt)):
        dt=tt[i]-tt[i-1] if i else 0.0; s=max(F[i],1e-3)/50.0; e=np.clip(e-s*k*wr[i]*dt,-s*cp,s*cn); out.append(e.sum()); V.append(np.sum(e**2/(2*s*k)))
    return np.array(out),np.array(V)
def load(tr):
    m=json.loads(str(z[tr+'/meta'])); hB=[str(x) for x in z[tr+'/armB_hdr']]; B=z[tr+'/armB'].astype(float); t0=z[tr+'/armB_t0']; thB=B[:,0]-t0[0]+t0[1]; qd=B[:,hB.index('actual_qd_5')]
    L=np.array(m['free_log'],float); c={kk:n for n,kk in enumerate(m['free_log_cols'])}; ev={n.split()[0]:tt for n,tt in m['events']}
    tt=np.arange(ev['contact'],ev['release'],0.002); tq=np.interp(tt,L[:,c['t']],L[:,c['tau_z']]); F=np.interp(tt,L[:,c['t']],L[:,c['fA']])
    def at(shift):
        w=np.interp(tt+shift,thB,qd); return w
    w=at(0); tl,V=law(tt,w,F); th=np.degrees(np.cumsum(w)*0.002)
    zc=[i for i in range(1,len(tt)) if tq[i-1]*tq[i]<0 and tt[i]-tt[0]>0.5]; zc=[i for j,i in enumerate(zc) if j==0 or tt[i]-tt[zc[j-1]]>0.2]
    return m,tt,tq,F,w,tl,V,th,zc,at
res={}
print("trial I spin half-swings | E_swing | loss measured | law: work, stored change, loss | zeta measured (-10ms, +10ms) | zeta law")
for tr in trs:
    m=json.loads(str(z[tr+'/meta'])); r=m['row']
    if r['block']!='P' or r['mode']!='rigid': continue
    m,tt,tq,F,w,tl,V,th,zc,at=load(tr); I=float(r['inertia'])
    if len(zc)<2: print(tr[-4:],I,r['spin'],'no full half swing'); continue
    a,b=zc[0],zc[-1]; n=len(zc)-1; dt=0.002; Es=0.5*I*np.max(np.abs(w[a:b]))**2
    lm=-np.sum((tq*w)[a:b])*dt; lw=-np.sum((tl*w)[a:b])*dt; dV=V[b]-V[a]; ll=lw-dV
    zs=[-np.sum((tq*at(s))[a:b])*dt/n/(2*np.pi*Es) for s in (-0.01,0.01)]
    res[tr]=(I,float(r['spin']),lm/n/(2*np.pi*Es),ll/n/(2*np.pi*Es),zs[0],zs[1])
    print(f"{tr[-4:]} {I} {r['spin']:>4s} {n} | {Es:.3f} | {lm:+.3f} | {lw:+.3f} {dV:+.3f} {ll:+.3f} | {res[tr][2]:+.3f} ({zs[0]:+.3f}, {zs[1]:+.3f}) | {res[tr][3]:+.3f}")
for I in (0.5,1.0,2.0):
    g=np.array([v for v in res.values() if v[0]==I]); print("I",I,"n",len(g),"zeta measured %.3f (range %.3f-%.3f; timing %.3f-%.3f)  zeta law %.3f"%(g[:,2].mean(),g[:,2].min(),g[:,2].max(),g[:,5].mean(),g[:,4].mean(),g[:,3].mean()))
g=np.array([v for v in res.values() if v[0]>=1]); print("I>=1: n",len(g),"zeta measured mean %.3f range %.3f-%.3f, law mean %.3f range %.3f-%.3f"%(g[:,2].mean(),g[:,2].min(),g[:,2].max(),g[:,3].mean(),g[:,3].min(),g[:,3].max()))
# figure: I = 1 and 2, trials that start at -15 deg/s (two each)
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9),sharey=True)
for a_,(I,ts) in zip(ax,((1.0,("trial_0515","trial_0518")),(2.0,("trial_0503","trial_0521")))):
    a_.axhline(0,color="#6b6b6b",lw=0.5)
    for j,tr in enumerate(ts):
        m,tt,tq,F,w,tl,V,th,zc,at=load(tr); s=slice(zc[0],zc[-1]+1); th0=th[s]-th[s][0]
        a_.plot(th0,tq[s],color=CO,lw=1.3,label="hardware" if j==0 else None); a_.plot(th0,tl[s],color=CO,lw=0.9,ls=(0,(3,2)),label="law, same motion" if j==0 else None)
    a_.plot(0,0,"o",ms=4,color=CO,mec="white",mew=0.7,zorder=5,label="start" if I==1.0 else None)
    a_.set_title(f"I = {I:.1f} kg·m²",fontsize=8); a_.set_xlabel("rim angle from the start of the swing (deg)"); a_.set_xlim(-0.4,9.9)
ax[0].set_ylabel("torque on the target (N·m)"); ax[0].set_ylim(-1.8,2.1); ax[0].set_yticks([-1,0,1,2])
kw=dict(fontsize=7,color="#333333",arrowprops=dict(arrowstyle="-",color="#888888",lw=0.6,shrinkA=1,shrinkB=2))
ax[0].annotate("out",xy=(3.0,-0.62),xytext=(2.2,-1.25),ha="center",**kw); ax[0].annotate("back, hardware",xy=(1.9,-0.25),xytext=(1.6,0.75),ha="center",**kw); ax[0].annotate("back, law",xy=(4.6,0.62),xytext=(6.6,1.35),ha="center",**kw)
ax[0].text(-0.14,1.06,"(a)",transform=ax[0].transAxes,fontweight="bold"); ax[1].text(-0.05,1.06,"(b)",transform=ax[1].transAxes,fontweight="bold")
fig.legend(*ax[0].get_legend_handles_labels(),loc="upper center",ncol=3,fontsize=7); fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig(os.path.join(FIG,"FigS4.png")); fig.savefig(os.path.join(FIG,"FigS4.pdf"))
