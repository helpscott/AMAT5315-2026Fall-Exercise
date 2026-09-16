#!/usr/bin/env python3
import pathlib,numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from common import rows,by_temp
ROOT=pathlib.Path(__file__).resolve().parents[1];E=ROOT/'evidence';E.mkdir(exist_ok=True)
fig,ax=plt.subplots(2,1,figsize=(6,7))
for name in ('window-l64','wolff-l64'):
 p=ROOT/'artifacts'/name/'series.jsonl';
 if not p.exists():continue
 d=by_temp(rows(p));ts=list(d); m=[np.mean(np.abs([r['M'] for r in d[t]])) for t in ts]; chi=[len(d[t])*(np.mean(np.array([r['M'] for r in d[t]])**2)-m[i]**2)/t for i,t in enumerate(ts)];ax[0].plot(ts,m,'o-',label=name);ax[1].plot(ts,chi,'o-',label=name)
ax[0].legend();ax[1].legend();ax[0].set_ylabel('mean |M|');ax[1].set_ylabel('cluster susceptibility');ax[1].set_xlabel('T');plt.tight_layout();plt.savefig(E/'magnetization-compare.png',dpi=150);plt.close()
