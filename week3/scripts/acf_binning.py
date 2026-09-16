#!/usr/bin/env python3
import pathlib,numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from common import rows,tau_int
ROOT=pathlib.Path(__file__).resolve().parents[1];E=ROOT/'evidence';E.mkdir(exist_ok=True)
fig,ax=plt.subplots(1,2,figsize=(10,4))
for T,c in [(2.3,'C0'),(3.0,'C1')]:
 p=ROOT/'artifacts'/f'T{T:.1f}'/'series.jsonl';
 if not p.exists():continue
 x=np.abs(np.array([r['M'] for r in rows(p) if abs(r['T']-T)<1e-6])[:5000]); x-=x.mean();v=np.dot(x,x)/len(x); rho=[1.0]+[np.dot(x[:-k],x[k:])/(len(x)-k)/v for k in range(1,min(500,len(x)-1))]; ax[0].plot(rho,label=f'T={T}',color=c)
 bs=[1,2,4,8,16,32,64,128,256,512,1024]; se=[]
 for b in bs:
  z=np.array([x[i:i+b].mean() for i in range(0,len(x)-b+1,b)]);se.append(z.std(ddof=1)/np.sqrt(len(z)))
 ax[1].plot(bs,se,'o-',label=f'T={T}',color=c)
ax[0].set_xlabel('lag');ax[0].set_ylabel('ACF');ax[1].set_xscale('log');ax[1].set_xlabel('block length');ax[1].set_ylabel('SE');ax[0].legend();ax[1].legend();plt.tight_layout();plt.savefig(E/'acf-binning.png',dpi=150);plt.close()
