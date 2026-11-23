# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""sweep_reversal.py - at each reversal of the rim in the sweeps (blocks H, H2): tangent stiffness just before and just after,
measured and from the law driven by the same motion. A slipping contact returns on a stiff branch; a soft spring returns on the same slope.

    python sweep_reversal.py path/to/contact_pack_s2.npz      (packed raw data of session 2, available from the authors)"""
import numpy as np, json, sys
z=np.load(sys.argv[1] if len(sys.argv)>1 else 'contact_pack_s2.npz',allow_pickle=True); trs=sorted({k.split('/')[0] for k in z.files})
Pp=np.array([18.226839128435657,7.576894038658541,2.5894049765172324,7.557897412443152e-27,0.5641941560541658,4.4045840701688315,1.8077901051997332,1.434108483334086,40.0]); k,cp,cn=Pp[:3],Pp[3:6],Pp[6:9]
def law(dt,wr,F):
    e=np.zeros(3); out=[]
    for i in range(len(wr)):
        s=max(F[i],1e-3)/50.0; e=np.clip(e-s*k*wr[i]*dt,-s*cp,s*cn); out.append(e.sum())
    return np.array(out)
def sm(x,n):
    return np.convolve(x,np.ones(n)/n,'same')
rows=[]; fits=[]
for t in trs:
    m=json.loads(str(z[t+'/meta'])); r=m['row']
    if r['block'] not in ('H','H2') or not m.get('contact'): continue
    hA=[str(x) for x in z[t+'/armA_hdr']]; A=z[t+'/armA'].astype(float); a0=z[t+'/armA_t0']; tA=A[:,0]-a0[0]+a0[1]
    hB=[str(x) for x in z[t+'/armB_hdr']]; B=z[t+'/armB'].astype(float); b0=z[t+'/armB_t0']; tB=B[:,0]-b0[0]+b0[1]
    ev={n.split()[0]:tt for n,tt in m['events']}; zA=np.array(m['axis_base'])
    i3=hA.index('actual_TCP_force_3'); i0=hA.index('actual_TCP_force_0')
    pre=(tA>ev['close_cmd']-0.3)&(tA<ev['close_cmd'])
    Tz=(A[:,i3:i3+3]-A[pre,i3:i3+3].mean(0))@zA; Fm=np.linalg.norm(A[:,i0:i0+3]-A[pre,i0:i0+3].mean(0),axis=1)
    # contact time: first time force rise > 5 N for 0.1 s
    k5=np.flatnonzero((Fm>5)&(tA>ev['close_cmd'])); tc=None
    for i in k5:
        j=np.searchsorted(tA,tA[i]+0.1)
        if j<len(tA) and (Fm[i:j]>5).all(): tc=tA[i]; break
    if tc is None: continue
    tt=np.arange(tc,ev['release'],0.002); tq=np.interp(tt,tA,sm(Tz,13)); F=np.interp(tt,tA,sm(Fm,25)); wB=np.interp(tt,tB,B[:,hB.index('actual_qd_5')])
    best=None
    for sw in (1,-1):
        for st in (1,-1):
            tl=law(0.002,sw*wB,F); e=np.sqrt(np.mean((st*tq-tl)[50:]**2))
            if best is None or e<best[0]: best=(e,sw,st,tl)
    e,sw,st,tl=best; w=sw*wB; tau=st*tq; th=np.degrees(np.cumsum(w)*0.002); fits.append((t,r['block'],r['spin'],sw,st,e))
    rev=[i for i in range(1,len(tt)) if w[i-1]*w[i]<0 and tt[i]-tt[0]>0.3]
    rev=[i for j,i in enumerate(rev) if j==0 or tt[i]-tt[rev[j-1]]>0.3]
    for i in rev:
        d=np.abs(th-th[i]); 
        a=np.flatnonzero((np.arange(len(tt))<i)&(d<=2.0)&(np.arange(len(tt))>i-600)); b=np.flatnonzero((np.arange(len(tt))>i)&(d<=2.0)&(np.arange(len(tt))<i+600))
        if len(a)<20 or len(b)<20 or d[a].max()<1.5 or d[b].max()<1.5: continue
        sl=lambda idx,y: -np.polyfit(np.radians(th[idx]),y[idx],1)[0]      # stiffness, N m/rad (law: tau = -K theta in stick)
        side='slip side' if w[i-1]>0 else 'wedge side'
        rows.append((t,r['block'],float(r['spin']),side,th[i],tau[i],F[i],sl(a,tau),sl(b,tau),sl(a,tl),sl(b,tl)))
import collections
print('law sign choices (w sign, torque sign):',collections.Counter((f[3],f[4]) for f in fits),' median fit rms %.2f N m over %d sweeps'%(np.median([f[5] for f in fits]),len(fits)))
R=rows
print('\nreversals found:',len(R))
print('side        amp   n  | angle at reversal  torque  force | stiffness before -> after reversal (N m/rad): MEASURED            LAW (same motion)')
for side in ('slip side','wedge side'):
    for amp in (12.,30.):
        g=[x for x in R if x[3]==side and abs(x[2])==amp]
        if not g: continue
        a=np.array([x[4:] for x in g]); med=np.median(a,0); q=lambda c:(np.percentile(a[:,c],25),np.percentile(a[:,c],75))
        print(f"{side:10s} {amp:4.0f} {len(g):3d} | {med[0]:+6.1f} deg  {med[1]:+5.2f} N m  {med[2]:4.0f} N | {med[3]:5.1f} ({q(3)[0]:.1f}-{q(3)[1]:.1f}) -> {med[4]:5.1f} ({q(4)[0]:.1f}-{q(4)[1]:.1f})      {med[5]:5.1f} -> {med[6]:5.1f}")
print('\nslip-side reversals, 12 deg/s sweeps, one line each: block, first direction, angle, torque, force, measured stiffness before -> after, law after')
for x in sorted([x for x in R if x[3]=='slip side' and abs(x[2])==12.],key=lambda x:(x[1],x[2],x[4])):
    print(f"  {x[0][-4:]} {x[1]:2s} {x[2]:+4.0f}  {x[4]:+5.1f} deg {x[5]:+5.2f} N m {x[6]:3.0f} N   {x[7]:5.1f} -> {x[8]:5.1f}   law {x[10]:5.1f}")
g=[x for x in R if x[3]=='slip side']
for lo,hi in ((0,5),(5,9),(9,14),(14,30)):
    a=np.array([(x[7],x[8],x[10]) for x in g if lo<=abs(x[4])<hi])
    if len(a): print(f"slip side, reversal angle {lo}-{hi} deg: n={len(a)}  measured after-reversal stiffness median {np.median(a[:,1]):.1f} (stiff return, >15 N m/rad, in {np.mean(a[:,1]>15)*100:.0f}% )  law {np.median(a[:,2]):.1f}")
