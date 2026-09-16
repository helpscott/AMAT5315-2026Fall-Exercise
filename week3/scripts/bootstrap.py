#!/usr/bin/env python3
import pathlib,numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import rows,by_temp
ROOT=pathlib.Path(__file__).resolve().parents[1];E=ROOT/'evidence';E.mkdir(exist_ok=True); rng=np.random.default_rng(2026)
def estimate(path,bs,reps=500):
 d=by_temp(rows(path)); ts=np.array(list(d)); vals=[]
 for _ in range(reps):
  chi=[]
  for T in ts:
   x=np.array([r['M'] for r in d[T]]); blocks=np.array_split(x,len(x)//bs); bm=np.array([z.mean() for z in blocks]); sample=rng.choice(bm,len(bm),replace=True); chi.append(len(x)*(np.mean(sample**2)-np.mean(np.abs(sample))**2)/T)
  i=int(np.argmax(chi)); lo=max(0,i-2); hi=min(len(ts),i+3); co=np.polyfit(ts[lo:hi],np.array(chi)[lo:hi],2); vals.append(-co[1]/(2*co[0]) if co[0]<0 else np.nan)
 return np.array(vals)
plt.figure(figsize=(6,4)); allv=[]
for bs in (2000,4000,8000):
 p=ROOT/'artifacts'/'window-l64'/'series.jsonl';
 if not p.exists():continue
 v=estimate(p,bs); v=v[np.isfinite(v)];allv.append(v);plt.hist(v,bins=25,histtype='step',label=f'block {bs}')
 if len(v): print(bs,np.mean(v),np.std(v))
plt.axvline(2.26919,color='k',ls='--');plt.xlabel('extrapolated Tc');plt.legend();plt.tight_layout();plt.savefig(E/'chi-bootstrap.png',dpi=150);plt.close()
