#!/usr/bin/env python3
import pathlib, numpy as np
from common import rows,by_temp,tau_int
ROOT=pathlib.Path(__file__).resolve().parents[1]; out=ROOT/'evidence'/'errors.txt'; out.parent.mkdir(exist_ok=True)
with open(out,'w') as f:
 for L in (32,64):
  p=ROOT/'artifacts'/f'window-l{L}'/'series.jsonl'
  if not p.exists(): continue
  for T,rr in by_temp(rows(p)).items():
   x=np.abs(np.array([r['M'] for r in rr])); n=len(x); naive=x.std(ddof=1)/np.sqrt(n); tau=tau_int(x); honest=naive*np.sqrt(2*tau); blocks=np.array_split(x,50); bse=np.std([b.mean() for b in blocks],ddof=1)/np.sqrt(50)
   f.write(f'L={L} T={T:.2f} mean_abs_M={x.mean():.6f} naive_se={naive:.6g} block_se={bse:.6g} ratio={bse/naive:.3f} tau_int={tau:.3f} honest_se={honest:.6g}\n')
