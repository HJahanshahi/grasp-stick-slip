# Author: Hadi Jahanshahi, York University (hadij@yorku.ca)
"""movie_S1.py - builds Movie S1: the rigid hand against the damping hand on the six-axis free body (trials 1401 and
1402), side by side, aligned at contact, with the recorded body spin below.

The camera recording is not in this repository. Before running, extract its frames into folders A/ and B/
(940 x 780 pixels, named by frame number) and keep movie_S1_sync.json, which gives, for each trial, the time offset between the
camera and the recorded data and the contact time. ffmpeg and the Inter font are needed."""
import numpy as np, json, subprocess, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image, ImageDraw, ImageFont
FPS=30000/1001; W,H=1920,1080
INTER=os.environ.get('INTER_FONT_DIR','/usr/share/fonts/opentype/inter/')
for f in os.listdir(INTER): fm.fontManager.addfont(INTER+f)
plt.rcParams['font.family']='Inter'
d=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','free_body','data','movie_trials.npz'),allow_pickle=True); ec=list(d['em_cols']); S=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'movie_S1_sync.json')))
CA,CB='#2a78d6','#eb6834'; INK='#1a1a1a'; MUT='#6b6b6b'
T0,T1=-3.0,5.5
tr={}
for k,t in (('A','1401'),('B','1402')):
    em=d[t+'_em']; tc=em[:,0]-S[t]['contact']; wz=np.degrees(em[:,ec.index('wz')])
    tr[k]=dict(tc=tc,wz=wz,rel=S[t]['release']-S[t]['contact'],vc=S[t]['off']+S[t]['contact'])
# static plot
PW,PH=1880,210
fig=plt.figure(figsize=(PW/100,PH/100),dpi=100); ax=fig.add_axes([0.055,0.25,0.925,0.68])
ax.axhline(0,color='#c9c9c9',lw=1); ax.axvline(0,color='#c9c9c9',lw=1,ls=(0,(4,3)))
for k,c,lab in (('A',CA,'rigid hand'),('B',CB,'damping hand')):
    q=tr[k]; m=(q['tc']>=T0)&(q['tc']<=T1); ax.plot(q['tc'][m],q['wz'][m],color=c,lw=2)
    ax.plot([q['rel']],[np.interp(q['rel'],q['tc'],q['wz'])],'o',ms=8,color=c,mec='white',mew=1.5,zorder=5)
ax.plot([-2.75],[10],'s',color=CA,ms=7); ax.text(-2.65,10,'rigid hand',color=INK,fontsize=13,va='center')
ax.plot([-2.75],[-3],'s',color=CB,ms=7); ax.text(-2.65,-3,'damping hand',color=INK,fontsize=13,va='center')
ax.text(-0.05,15,'contact',color=MUT,fontsize=12,va='center',ha='right'); ax.text(4.1,11,'dots mark the release',color=MUT,fontsize=12,va='center')
ax.set_xlim(T0,T1); ax.set_ylim(-29,23); ax.set_yticks([-25,0,20])
ax.set_xlabel('time from contact (s)',fontsize=13,color=INK,labelpad=2); ax.set_ylabel('body spin (deg/s)',fontsize=13,color=INK)
for s in ('top','right'): ax.spines[s].set_visible(False)
for s in ('left','bottom'): ax.spines[s].set_color('#9a9a9a')
ax.tick_params(colors=MUT,labelsize=12,length=3)
fig.savefig('plot.png',facecolor='white'); 
x0=ax.transData.transform((T0,0))[0]; x1=ax.transData.transform((T1,0))[0]; yb,yt=ax.transData.transform((0,-29))[1],ax.transData.transform((0,23))[1]
plot=Image.open('plot.png').convert('RGB')
F=lambda n,s:ImageFont.truetype(INTER+n,s)
f_lab,f_sm,f_ti=F('Inter-SemiBold.otf',34),F('Inter-Regular.otf',26),F('Inter-Medium.otf',26)
def frame(tc,mode):
    im=Image.new('RGB',(W,H),'white'); dr=ImageDraw.Draw(im)
    for k,x in (('A',15),('B',965)):
        n=int(round((tr[k]['vc']+tc)*FPS)); im.paste(Image.open('%s/%05d.jpg'%(k,n)),(x,64))
    dr.rectangle([15,12,29,46],fill=CA); dr.text((42,8),'Rigid hand',font=f_lab,fill=INK)
    dr.rectangle([965,12,979,46],fill=CB); dr.text((992,8),'Damping hand',font=f_lab,fill=INK)
    s='%s   t = %+.2f s'%(mode,tc); w=dr.textlength(s,font=f_ti); dr.text((W-15-w,14),s,font=f_ti,fill=MUT)
    im.paste(plot,(20,856))
    cx=20+x0+(tc-T0)/(T1-T0)*(x1-x0); dr.line([cx,856+PH-yt,cx,856+PH-yb],fill=INK,width=3)
    return im
seq=[(t,'real time') for t in np.arange(T0,T1,1/FPS)]+[(t,'3× slower') for t in np.arange(-0.3,2.7,1/FPS/3)]
if __name__=='__main__':
    import sys
    if len(sys.argv)>1:
        for i,(t,md) in enumerate([(-1.0,'real time'),(0.75,'3× slower'),(4.0,'real time')]): frame(t,md).save('chk%d.jpg'%i,quality=88)
        sys.exit()
    p=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','%dx%d'%(W,H),'-r','30000/1001','-i','-','-c:v','libx264','-preset','slow','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart','Movie_S1.mp4'],stdin=subprocess.PIPE)
    for t,md in seq: p.stdin.write(frame(t,md).tobytes())
    p.stdin.close(); p.wait(); print(len(seq))
