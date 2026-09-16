#!/usr/bin/env python3
import pathlib,numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import rows
ROOT=pathlib.Path(__file__).resolve().parents[1]; E=ROOT/'evidence';E.mkdir(exist_ok=True)
fig,ax=plt.subplots(2,1,figsize=(7,5),sharex=True)
for a,L,T in zip(ax,[64,64],[2.3,3.0]):
 p=ROOT/'artifacts'/f'T{T:.1f}'/'series.jsonl';
 if not p.exists(): continue
 rr=[r for r in rows(p) if abs(r['T']-T)<1e-6][:2000]; a.plot([r['sweep'] for r in rr],[abs(r['M']) for r in rr]);a.set_ylabel(f'|M| T={T}')
ax[-1].set_xlabel('recorded sweep');plt.tight_layout();plt.savefig(E/'trace.png',dpi=150);plt.close()
