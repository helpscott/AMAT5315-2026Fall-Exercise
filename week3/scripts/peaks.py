#!/usr/bin/env python3
import pathlib, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import rows,by_temp
ROOT=pathlib.Path(__file__).resolve().parents[1]; E=ROOT/'evidence'; E.mkdir(exist_ok=True)
def metrics(path):
 d=by_temp(rows(path)); ts=np.array(list(d)); ma=np.array([np.mean(np.abs([r['M'] for r in d[t]])) for t in ts]); chi=np.array([len(d[t])*(np.mean(np.array([r['M'] for r in d[t]])**2)-np.mean(np.abs([r['M'] for r in d[t]]))**2)/t for t in ts]); return ts,ma,chi
coarse={}; window={}
for L in (32,64):
 p=ROOT/'artifacts'/f'coarse-l{L}'/'series.jsonl'; q=ROOT/'artifacts'/f'window-l{L}'/'series.jsonl'
 if p.exists(): coarse[L]=metrics(p)
 if q.exists(): window[L]=metrics(q)
plt.figure(figsize=(6,4));
for L,(t,m,c) in sorted(coarse.items()): plt.plot(t,m,'o-',label=f'L={L}')
t=np.linspace(1.5,3.5,300); ons=np.where(t<2.26919,(1-np.sinh(2/t)**-4)**.125,0); plt.plot(t,ons,'k--',label='Onsager'); plt.xlabel('T');plt.ylabel('mean |M|');plt.legend();plt.tight_layout();plt.savefig(E/'magnetization.png',dpi=150);plt.close()
plt.figure(figsize=(6,4));
for L,(t,m,c) in sorted(coarse.items()): plt.plot(t,c,'o-',label=f'L={L}')
plt.axvline(2.26919,color='k',ls='--');plt.xlabel('T');plt.ylabel('susceptibility');plt.legend();plt.tight_layout();plt.savefig(E/'susceptibility.png',dpi=150);plt.close()
with open(E/'peaks.txt','w') as f:
 peaks={}
 for L,(t,m,c) in sorted(window.items()):
  i=int(np.argmax(c)); lo=max(0,i-2); hi=min(len(t),i+3); tt=t[lo:hi]; cc=c[lo:hi]
  if len(tt)>=3:
   co=np.polyfit(tt,cc,2); tp=float(-co[1]/(2*co[0])) if co[0]!=0 else float(t[i]);
  else: tp=float(t[i])
  peaks[L]=tp; f.write(f'L={L} T_peak={tp:.6f} max_chi={c[i]:.6f}\n')
 if len(peaks)==2: f.write(f'Tc_extrapolated={2*peaks[64]-peaks[32]:.6f}\n')
