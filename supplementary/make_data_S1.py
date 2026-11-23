# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""make_data_S1.py - builds Data S1: one CSV of plotted values for each figure, and an index of the series.
It runs the figure scripts with plot_recorder.py switched on, then writes supplementary/Data_S1/.

    python make_data_S1.py
"""
import json, csv, os, sys, subprocess, tempfile, shutil, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.join(HERE,".."); D=tempfile.mkdtemp(); O=os.path.join(HERE,"Data_S1"); os.makedirs(O,exist_ok=True)
env=dict(os.environ,DUMP_DIR=D,PYTHONPATH=HERE+os.pathsep+os.environ.get("PYTHONPATH",""))
ONE="import plot_recorder,runpy,sys;sys.path.insert(0,'.');runpy.run_path('fig3_law_schematic.py',run_name='__main__');import figs;[getattr(figs,f)() for f in ('fig2','fig3','fig4_angle','fig5_loops','fig6_validation')];runpy.run_path('closed_loop.py',run_name='__main__')"
os.makedirs(os.path.join(ROOT,"one_axis","out","figs"),exist_ok=True)
subprocess.run([sys.executable,"-c",ONE],check=True,cwd=os.path.join(ROOT,"one_axis"),env=env,stdout=subprocess.DEVNULL)
for s in ("fig1.py","fig7_figS3_tableS4.py","figS4_grasp_loss.py"):
    subprocess.run([sys.executable,"-c","import plot_recorder,runpy,sys;sys.argv=['%s'];runpy.run_path('%s',run_name='__main__')"%(s,s)],check=True,cwd=os.path.join(ROOT,"free_body"),env=env,stdout=subprocess.DEVNULL)
HAND={'#d55e00':'rigid hand','#0072b2':'damping hand'}
cm=plt.cm.PuOr; RATE={('#9a9a9a' if w==0 else matplotlib.colors.to_hex(cm(0.5+0.47*np.sign(w)*(0.25+0.75*abs(w)/30)))):w for w in range(-40,41)}
SPEED={'#1b068d':40,'#9613a1':100,'#d9586a':170,'#f9983e':255}
REF={'#555555','#6b6b6b','#8c8c8c','#000000','#1a1a1a','#444444'}
def hw(s): return 'hardware' if s['ls'] in('-','solid') else 'law'
def hand_line(s): return '%s, %s'%(HAND.get(s['color'],s['color']),hw(s))
def hand_mark(s,what):
    c=s['color'].split()[0]; return '%s, %s, %s starting spin'%(what,HAND.get(c,c),'negative' if 'open' in s['color'] else 'positive')
def lab(s): return None if s['label'].startswith('_') else s['label']
def f1c(s):
    return {'#b9561c':'law at 50 N, wedging','#5b3a94':'law at 50 N, slip','#1c1c1c':'law at 50 N, stick','#8c8c8c':'law at 25 N'}.get(s['color'])
def f1d(s): return (lab(s) or HAND.get(s['color'],''))+(', law' if s['kind']=='line' else ', fingers open')
def f3a(s):
    if s['color'] in RATE and s['ls']=='-': return 'measured, rim rate %+d deg/s'%RATE[s['color']]
    if s['ls']=='--': return 'law, one hold'
def f3b(s):
    if s['kind']=='scatter':
        f3b.k=getattr(f3b,'k',0)+1; return 'holds, grip setting %d'%(40,100,180)[f3b.k-1]
    return 'median at each rim rate'
def f4(s): return {'#2c7fb8':'measured, slip direction first','#d95f0e':'measured, wedging direction first','#08306b':'law, slip direction first','#7f2704':'law, wedging direction first'}.get(s['color'])
def f5(s): return lab(s).split(', n')[0] if s['kind']=='scatter' else None
def f6abc(s): return hand_line(s) if s['kind']=='line' else HAND.get(s['color'].split()[0],'')+', release'
def f6d(s):
    if s['kind']=='markers': return hand_mark(s,'trial (hardware)')
    return 'mean, '+hand_line(s)
def f6e(s): return HAND.get(s['color'],'')+', trials' if s['kind']=='scatter' else None
def f7a(s): return hand_line(s) if s['kind']=='line' else HAND.get(s['color'].split()[0],'')+', fingers open'
def f7b(s): return hand_mark(s,'trial') if s['kind']=='markers' else None
def f7cd(s):
    if s['kind']=='markers': return hand_mark(s,'trial (hardware)')
    if s['kind']=='line': return 'mean, '+hand_line(s)
def s1(s): return lab(s) or ('torque on the hand' if s['color']=='#0072b2' else None)
def s2a(s): return lab(s)
def s2b(s): return ('trials, speed setting %d'%SPEED[s['color']] if s['color'] in SPEED else 'trials') if s['kind']=='scatter' else 'proportional fit'
def s2c(s): return 'trial, speed setting %d'%SPEED[s['color']] if s['color'] in SPEED else 'fit over 3 to 12 units'
def s3(s): return ('rigid hand, '+hw(s)) if s['kind']=='line' else 'fingers open'
def s4(s): return ('hardware' if s['ls']=='-' else 'law, same motion') if s['kind']=='line' else 'start of the half swing'
def s5b(s): return 'loading cycle' if s['ls']=='-' else 'first loading'
FIG=[ # file, source json, {axis:(panel, x, y, namer, keep reference lines)}
('Fig1',  'Fig1', {2:('C','rim angle (deg)','torque (N·m)',f1c),3:('D','time from contact (s)','target spin (deg/s)',f1d)}),
('Fig3',  'fig4_torque_angle', {0:('A','rim angle turned since contact (deg)','torque on the hand (N·m)',f3a),1:('B','rim angle turned since contact (deg)','grasp force rise (N)',f3b)}),
('Fig4',  'fig5_hysteresis_angle', {0:('A','rim angle turned since contact (deg)','torque on the hand (N·m)',f4),1:('B','rim angle turned since contact (deg)','torque on the hand (N·m)',f4)}),
('Fig5',  'fig6_law_validation', {0:('','measured torque at the end of the hold (N·m)','torque predicted by the law (N·m)',f5)}),
('Fig6',  'fig7_closed_loop', dict([(i,('A, I = %s kg·m²'%v,'time from contact (s)','target spin (deg/s)',f6abc)) for i,v in zip((0,1,2),('0.5','1','2'))]+[(i,('B, I = %s kg·m²'%v,'time from contact (s)','torque on the target (N·m)',f6abc)) for i,v in zip((3,4,5),('0.5','1','2'))]+[(i,('C, I = %s kg·m²'%v,'time from contact (s)','hand angle (deg)',f6abc)) for i,v in zip((6,7,8),('0.5','1','2'))]+[(9,('D','target inertia (kg·m²)','spin after release (deg/s)',f6d)),(10,('E','release impulse divided by inertia (deg/s)','spin change at release (deg/s)',f6e))])),
('Fig7',  'Fig7', dict([(i,('A, I = %s kg·m²'%v,'time from contact (s)','target spin (deg/s)',f7a)) for i,v in zip((0,1,2),('0.58','1.16','2.32'))]+[(3,('B','measured stop time (s)','law stop time (s)',f7b)),(4,('C','target inertia (kg·m²)','rebound while held (deg/s)',f7cd)),(5,('D','target inertia (kg·m²)','spin at release (deg/s)',f7cd))])),
('FigS1', 'fig2_time_histories', {0:('A','time from contact (s)','force rise (N)',s1),1:('B','time from contact (s)','torque on the hand (N·m)',s1),2:('C','time from contact (s)','rim rate (deg/s)',s1),3:('D','time from contact (s)','finger position (hand units)',s1),4:('E','time from contact (s)','tactile deviation (1000 counts)',s1)}),
('FigS2', 'fig3_closure_map', {0:('A','time from contact (s)','force rise (N)',s2a),1:('B','finger speed before contact (units/s)','loading rate (N/s)',s2b),2:('C','finger travel since contact (units)','force rise (N)',s2c)}),
('FigS3', 'FigS3', {0:('A, I = 1.16 kg·m²','time from contact (s)','target spin (deg/s)',s3),1:('A, I = 2.32 kg·m²','time from contact (s)','target spin (deg/s)',s3),2:('B, I = 1.16 kg·m²','time from contact (s)','target spin (deg/s)',s3),3:('B, I = 2.32 kg·m²','time from contact (s)','target spin (deg/s)',s3)}),
('FigS4', 'FigS4', {0:('A','rim angle from the start of the half swing (deg)','torque on the target (N·m)',s4),1:('B','rim angle from the start of the half swing (deg)','torque on the target (N·m)',s4)}),
('FigS5', 'fig3_law_schematic', {1:('B','rim angle (arbitrary units)','element torque (arbitrary units)',s5b)}),
]
summary=[]; idx=[]
for name,src,axes in FIG:
    J=json.load(open(os.path.join(D,src+'.json'))); rows=0
    with open(os.path.join(O,'DataS1_%s.csv'%name),'w',newline='',encoding='utf-8') as fh:
        w=csv.writer(fh); w.writerow(['panel','series_no','x','y'])
        for ai,(panel,xq,yq,namer) in axes.items():
            n=0; pl=panel.split(',')[0] if panel else '-'; sub=panel.split(', ',1)[1] if ', ' in panel else ''
            for s in J[ai]['series']:
                if s['kind'] in('bars','errorbar'): continue
                c=s['color'].split()[0] if s['color'] else ''
                nm=namer(s) or lab(s)
                if s['kind']=='line' and len(s['x'])<=2 and c in REF: continue          # axis and reference lines
                if nm is None:
                    if not(s['kind']=='line' and c in REF): summary.append(('UNNAMED',name,panel,s['kind'],s['color'],s['ls'],len(s['x'])))
                    continue
                n+=1; k=0; sid='%s%d'%(pl if not sub else pl+str(list(axes).index(ai)%3+1)+'-',n) 
                for x,y in zip(s['x'],s['y']):
                    if x is None or y is None: continue
                    if xq.startswith('target inertia'): x=((0.5,1,2) if name=='Fig6' else (0.29,0.58,1.16,2.32))[int(round(x))]
                    w.writerow([panel,n,'%.6g'%x,'%.6g'%y]); rows+=1; k+=1
                idx.append([name.replace('Fig','Fig. '),panel,n,nm,xq,yq,k])
    summary.append((name,rows))
with open(os.path.join(O,'DataS1_index.csv'),'w',newline='',encoding='utf-8-sig') as fh:
    w=csv.writer(fh); w.writerow(['figure','panel','series_no','series','x_quantity','y_quantity','points']); w.writerows(idx)
for s in summary: print(s)
shutil.rmtree(D)
