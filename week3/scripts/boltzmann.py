#!/usr/bin/env python3
import pathlib,numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import rows
ROOT=pathlib.Path(__file__).resolve().parents[1]; E=ROOT/'evidence';E.mkdir(exist_ok=True)
def hist(path): return np.array([round(r['E']*4096/40)*40 for r in rows(path)])
a=hist(ROOT/'artifacts/T3.0/series.jsonl'); b=hist(ROOT/'artifacts/T3.1/series.jsonl'); bins=np.arange(min(a.min(),b.min()),max(a.max(),b.max())+40,40); ca,_=np.histogram(a,bins); cb,_=np.histogram(b,bins); x=[];y=[]
for i in range(len(ca)):
 if ca[i]>=5 and cb[i]>=5: x.append((bins[i]+bins[i+1])/2); y.append(np.log(cb[i]/ca[i]))
plt.figure(figsize=(6,4));plt.scatter(x,y);plt.xlabel('total energy bin');plt.ylabel('log P(3.1)/P(3.0)');plt.tight_layout();plt.savefig(E/'boltzmann.png',dpi=150);plt.close()
