#!/usr/bin/env python3
import pathlib,numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from common import rows,by_temp,tau_int
ROOT=pathlib.Path(__file__).resolve().parents[1];E=ROOT/'evidence';E.mkdir(exist_ok=True);plt.figure(figsize=(6,4))
for L in (32,64):
 p=ROOT/'artifacts'/f'window-l{L}'/'series.jsonl';
 if not p.exists():continue
 d=by_temp(rows(p));plt.plot(list(d),[tau_int(np.abs(np.array([r['M'] for r in z]))) for z in d.values()],'o-',label=f'L={L}')
plt.yscale('log');plt.xlabel('T');plt.ylabel('tau_int');plt.legend();plt.tight_layout();plt.savefig(E/'tau.png',dpi=150);plt.close()
