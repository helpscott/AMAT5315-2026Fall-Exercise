#!/usr/bin/env python3
import pathlib,numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import rows,by_temp,tau_int
ROOT=pathlib.Path(__file__).resolve().parents[1]; E=ROOT/'evidence';E.mkdir(exist_ok=True)
plt.figure(figsize=(6,4))
for name,fac in [('window-l64',1.0),('wolff-l64',None)]:
 p=ROOT/'artifacts'/name/'series.jsonl'
 if not p.exists():continue
 d=by_temp(rows(p)); ts=[]; vals=[]
 for T,rr in d.items():
  x=np.abs(np.array([r['M'] for r in rr])); tau=tau_int(x)
  if fac is None: tau=tau*np.mean([r.get('cluster_size',1) for r in rr])/(64*64)
  ts.append(T); vals.append(tau)
 plt.plot(ts,vals,'o-',label=name)
plt.yscale('log');plt.xlabel('T');plt.ylabel('work-normalized tau');plt.legend();plt.tight_layout();plt.savefig(E/'tau-compare.png',dpi=150);plt.close()
